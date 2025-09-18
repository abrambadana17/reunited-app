from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash, send_from_directory
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail  
from resnet_model import extract_features, auto_match

import MySQLdb.cursors
import re
import numpy as np
from datetime import datetime
import os
import uuid
from PIL import Image

app = Flask(__name__)
app.secret_key = '071322'  # Change this to a random secret key

import os

# MySQL Configuration (Railway + local fallback)
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'reunited_db')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', 3306))

# Email Sender (Gmail via App Password)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'reunited.uc@gmail.com')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = (
    os.getenv('MAIL_DEFAULT_NAME', 'Reunited Team'),
    os.getenv('MAIL_USERNAME', 'reunited.uc@gmail.com')
)

mail = Mail(app)


# File upload configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads/profile_pictures'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

mysql = MySQL(app)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image_path, max_size=(400, 400)):
    """Resize image to max_size while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(image_path, optimize=True, quality=85)
    except Exception as e:
        print(f"Error resizing image: {e}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# Add this function to calculate and update dashboard stats
def update_dashboard_stats(user_id, cursor, mysql):
    """Update dashboard statistics for a user"""
    # Count total items posted by user
    cursor.execute("""SELECT COUNT(*) as total_posted, COUNT(CASE WHEN type = 'lost' THEN 1 END) as lost_posted, COUNT(CASE WHEN type = 'found' THEN 1 END) as found_posted FROM items WHERE user_id = %s""", (user_id,))
    item_counts = cursor.fetchone()
    
    # Count claimed/returned items (items that are resolved)
    cursor.execute("""SELECT COUNT(*) as total_claimed FROM items WHERE user_id = %s AND (status = 'Claimed' OR status = 'Returned')""", (user_id,))
    claimed_count = cursor.fetchone()
    
    # Count total activity (matches involving user's items)
    cursor.execute("""SELECT COUNT(*) as total_activity FROM ai_matches m JOIN items i ON (m.lost_item_id = i.id OR m.found_item_id = i.id) WHERE i.user_id = %s""", (user_id,))
    activity_count = cursor.fetchone()
    
    # Count unread notifications
    cursor.execute("""SELECT COUNT(*) as unread_notifications FROM notifications WHERE user_id = %s AND is_read = 0""", (user_id,))
    notifications_count = cursor.fetchone()
    
    # Check if stats exist for user
    cursor.execute("SELECT id FROM dashboard_stats WHERE user_id = %s", (user_id,))
    existing_stats = cursor.fetchone()
    
    if existing_stats:
        # Update existing stats
        cursor.execute("""UPDATE dashboard_stats SET total_posted = %s, total_claimed = %s, total_activity = %s, last_updated = NOW() WHERE user_id = %s""", (item_counts['total_posted'], claimed_count['total_claimed'], activity_count['total_activity'], user_id))
    else:
        # Insert new stats
        cursor.execute("""INSERT INTO dashboard_stats (user_id, total_posted, total_claimed, total_activity, last_updated) VALUES (%s, %s, %s, %s, NOW())""", (user_id, item_counts['total_posted'], claimed_count['total_claimed'], activity_count['total_activity']))
    
    mysql.connection.commit()
    
    return {
        'total_posted': item_counts['total_posted'],
        'lost_posted': item_counts['lost_posted'],
        'found_posted': item_counts['found_posted'],
        'total_claimed': claimed_count['total_claimed'],
        'total_returned': claimed_count['total_claimed'],  # Using claimed as returned
        'total_activity': activity_count['total_activity']
    }

# Replace your existing dashboard route with this:
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Update and get dashboard stats
    stats = update_dashboard_stats(session['user_id'], cursor, mysql)
    
    # Get recent activity (last 5 items posted by user)
    cursor.execute("""SELECT i.title, i.type, i.created_at, img.file_path as image_path FROM items i LEFT JOIN images img ON i.id = img.item_id WHERE i.user_id = %s ORDER BY i.created_at DESC LIMIT 5""", (session['user_id'],))
    recent_items = cursor.fetchall()
    
    # Fix image paths for display
    for item in recent_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            if item['image_path'].startswith("static/"):
                item['image_path'] = item['image_path'][7:]
    
    cursor.execute("""SELECT f.message, f.rating, f.created_at, u.first_name, u.last_name, u.profile_picture FROM feedback f JOIN users u ON f.user_id = u.id WHERE f.is_public = 1 ORDER BY f.created_at DESC LIMIT 6""")
    recent_feedback = cursor.fetchall()
    
    cursor.close()
    
    return render_template('dashboard.html', active_page='dashboard', user=session, stats=stats, recent_items=recent_items, recent_feedback=recent_feedback)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'user_id' not in session:
        flash('Please log in to submit feedback.', 'danger')
        return redirect(url_for('login_page'))
    
    try:
        message = request.form.get('message', '').strip()
        rating = request.form.get('rating')
        is_public = 1 if request.form.get('is_public') else 0
        
        if not message:
            flash('Please provide feedback message.', 'danger')
            return redirect(url_for('dashboard'))
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Insert feedback
        cursor.execute("""INSERT INTO feedback (user_id, message, rating, is_public, created_at) VALUES (%s, %s, %s, %s, NOW())""", (session['user_id'], message, rating if rating else None, is_public))
        
        mysql.connection.commit()
        cursor.close()
        
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Error submitting feedback: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

# Serve uploaded files
@app.route('/uploads/profile_pictures/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        first_name = data.get('firstName', '').strip()
        last_name = data.get('lastName', '').strip()
        email = data.get('email', '').strip().lower()
        phone = data.get('phone', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validation
        errors = []
        
        if not first_name:
            errors.append("First name is required")
        if not last_name:
            errors.append("Last name is required")
        if not email:
            errors.append("Email is required")
        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            errors.append("Invalid email format")
        if not phone:
            errors.append("Phone number is required")
        elif not re.match(r'^[\d\s\+\-()]{10,}$', phone):
            errors.append("Invalid phone number format")
        if not password:
            errors.append("Password is required")
        elif len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        elif not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        elif not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        elif not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        if password != confirm_password:
            errors.append("Passwords do not match")
        
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Check if user already exists
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            return jsonify({'success': False, 'errors': ['Email already registered']}), 400
        
        # Hash password and create user
        hashed_password = generate_password_hash(password)
        
        cursor.execute('''INSERT INTO users (first_name, last_name, email, phone, password) VALUES (%s, %s, %s, %s, %s)''', (first_name, last_name, email, phone, hashed_password))
        
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Account created successfully! You can now log in.'
        }), 201
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Validation
        if not username or not password:
            return jsonify({'success': False, 'errors': ['Username and password are required']}), 400
        
        # Check if user exists (by email or username)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND is_active = TRUE', (username.lower(),))
        user = cursor.fetchone()
        cursor.close()
        
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'success': False, 'errors': ['Invalid credentials']}), 401
        
        # Create session
        session['user_id'] = user['id']
        session['first_name'] = user['first_name']
        session['last_name'] = user['last_name']
        session['email'] = user['email']
        session['full_name'] = f"{user['first_name']} {user['last_name']}"
        session['phone'] = user['phone']
        session['profile_picture'] = user['profile_picture']
        
        return jsonify({
            'success': True, 
            'message': 'Login successful!',
            'user': {
                'id': user['id'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'email': user['email'],
                'phone': user['phone'],
                'profile_picture': user['profile_picture']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

@app.route('/api/claim', methods=['POST'])
def claim_item():
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        data = request.get_json()
        notification_id = data.get('notification_id')
        item_id = data.get('item_id')
        
        if not all([notification_id, item_id]):
            return jsonify({'success': False, 'errors': ['Missing required fields']}), 400
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""SELECT n.*, m.id as match_id, m.lost_item_id, m.found_item_id, lost.user_id as lost_user_id, lost.title as lost_title, lost.status as lost_status, found.user_id as found_user_id, found.title as found_title, found.status as found_status FROM notifications n JOIN ai_matches m ON n.match_id = m.id JOIN items lost ON m.lost_item_id = lost.id JOIN items found ON m.found_item_id = found.id WHERE n.id = %s""", (notification_id,))
        
        notification_data = cursor.fetchone()
        if not notification_data:
            return jsonify({'success': False, 'errors': ['Notification not found']}), 404
        
        user_id = session['user_id']
        
        if user_id == notification_data['lost_user_id']:
            action_type = 'claimed'
            target_item_id = notification_data['lost_item_id']
            new_status = 'Claimed'
        elif user_id == notification_data['found_user_id']:
            action_type = 'returned'
            target_item_id = notification_data['found_item_id']
            new_status = 'Returned'
        else:
            return jsonify({'success': False, 'errors': ['Not authorized for this action']}), 403
        
        cursor.execute("SELECT status FROM items WHERE id = %s", (target_item_id,))
        current_item = cursor.fetchone()
        
        if (action_type == 'claimed' and current_item['status'] == 'Claimed') or (action_type == 'returned' and current_item['status'] == 'Returned'):
            return jsonify({'success': False, 'errors': ['Already marked as ' + action_type]}), 400
        
        cursor.execute("UPDATE items SET status = %s WHERE id = %s", (new_status, target_item_id))
        cursor.execute("UPDATE notifications SET is_read = 1 WHERE id = %s", (notification_id,))
        mysql.connection.commit()
        
        cursor.execute("""SELECT UPPER(lost.status) as lost_status, UPPER(found.status) as found_status FROM ai_matches m JOIN items lost ON m.lost_item_id = lost.id JOIN items found ON m.found_item_id = found.id WHERE m.id = %s""", (notification_data['match_id'],))
        
        match_status = cursor.fetchone()
        both_completed = (match_status['lost_status'] == 'CLAIMED' and match_status['found_status'] == 'RETURNED')
        
        if both_completed:
            cursor.execute("""SELECT c.id FROM claims c WHERE c.lost_user_id = %s AND c.found_user_id = %s AND c.item_lost_id = %s AND c.item_found_id = %s""", (notification_data['lost_user_id'], notification_data['found_user_id'], notification_data['lost_item_id'], notification_data['found_item_id']))
            
            existing_claim = cursor.fetchone()
            
            if not existing_claim:
                cursor.execute("""INSERT INTO claims (lost_user_id, found_user_id, item_lost_id, item_found_id, claim_date) VALUES (%s, %s, %s, %s, NOW())""", (notification_data['lost_user_id'], notification_data['found_user_id'], notification_data['lost_item_id'], notification_data['found_item_id']))
                
                claim_id = cursor.lastrowid
                
                cursor.execute("""UPDATE ai_matches SET status = 'resolved' WHERE id = %s""", (notification_data['match_id'],))
                
                mysql.connection.commit()
        
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': f'Item marked as {action_type} successfully!',
            'both_completed': both_completed,
            'action_type': action_type
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/claim-status/<int:notification_id>')
def get_claim_status(notification_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""SELECT match_id FROM notifications WHERE id = %s""", (notification_id,))
        
        notification = cursor.fetchone()
        if not notification:
            return jsonify({'success': False, 'errors': ['Notification not found']}), 404
        
        match_id = notification['match_id']
        
        cursor.execute("""SELECT m.status as match_status, lost.user_id as lost_user_id, lost.status as lost_item_status, found.user_id as found_user_id, found.status as found_item_status, u1.first_name as lost_first_name, u1.last_name as lost_last_name, u2.first_name as found_first_name, u2.last_name as found_last_name FROM ai_matches m JOIN items lost ON m.lost_item_id = lost.id JOIN users u1 ON lost.user_id = u1.id JOIN items found ON m.found_item_id = found.id JOIN users u2 ON found.user_id = u2.id WHERE m.id = %s""", (match_id,))
        
        match_data = cursor.fetchone()
        cursor.close()
        
        if not match_data:
            return jsonify({'success': False, 'errors': ['Match not found']}), 404
        
        claim_status = {
            'claimed': None,
            'returned': None,
            'both_completed': False
        }
        
        if match_data['lost_item_status'] == 'Claimed':
            claim_status['claimed'] = {
                'user_id': match_data['lost_user_id'],
                'name': f"{match_data['lost_first_name']} {match_data['lost_last_name']}",
                'status': 'resolved' if match_data['match_status'] == 'resolved' else 'pending'
            }
        
        if match_data['found_item_status'] == 'Returned':
            claim_status['returned'] = {
                'user_id': match_data['found_user_id'],
                'name': f"{match_data['found_first_name']} {match_data['found_last_name']}",
                'status': 'resolved' if match_data['match_status'] == 'resolved' else 'pending'
            }
        
        claim_status['both_completed'] = match_data['match_status'] == 'resolved'
        
        return jsonify({'success': True, 'claim_status': claim_status}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/notifications')
def notifications():
    if 'user_id' not in session:
        flash("Please log in to view notifications.", "danger")
        return redirect(url_for("login_page"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT n.id, n.item_id, n.match_id, n.type, n.message, n.is_read, n.sent_at, i.type as item_type, i.user_id as item_owner_id FROM notifications n LEFT JOIN items i ON n.item_id = i.id WHERE n.user_id = %s ORDER BY n.sent_at DESC""", (session['user_id'],))
    notifications = cursor.fetchall()
    cursor.close()

    return render_template("notifications.html", notifications=notifications, active_page="notifications")

@app.context_processor
def inject_unread_notifications():
    if 'user_id' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT COUNT(*) AS cnt FROM notifications WHERE user_id = %s AND is_read = 0", (session['user_id'],))
        result = cursor.fetchone()
        cursor.close()
        return dict(unread_count=result['cnt'])
    return dict(unread_count=0)

@app.route('/api/unread_count')
def unread_count_api():
    if 'user_id' not in session:
        return {"count": 0}

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT COUNT(*) AS cnt FROM notifications WHERE user_id = %s AND is_read = 0", (session['user_id'],))
    result = cursor.fetchone()
    cursor.close()
    return {"count": result['cnt']}

@app.route('/notifications/read/<int:notif_id>')
def mark_notification_read(notif_id):
    if 'user_id' not in session:
        return redirect(url_for("login_page"))

    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE notifications SET is_read = 1 WHERE id=%s AND user_id=%s", (notif_id, session['user_id']))
    mysql.connection.commit()
    cursor.close()
    return redirect(url_for("notifications"))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')

        first_name, last_name = "", ""
        if " " in full_name:
            first_name, last_name = full_name.split(" ", 1)
        else:
            first_name = full_name
            last_name = ""

        # Handle profile picture upload
        profile_picture_filename = session.get('profile_picture')
        
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename != '' and allowed_file(file.filename):
                # Delete old profile picture if it exists
                if profile_picture_filename:
                    old_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_picture_filename)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                # Generate unique filename
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                profile_picture_filename = f"user_{session['user_id']}_{uuid.uuid4().hex[:8]}.{file_extension}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_picture_filename)
                
                # Save and resize image
                file.save(file_path)
                resize_image(file_path)

        # Update SQL
        if password:
            hashed_password = generate_password_hash(password)
            cursor.execute('''UPDATE users SET first_name=%s, last_name=%s, email=%s, phone=%s, password=%s, profile_picture=%s WHERE id=%s''', (first_name, last_name, email, phone, hashed_password, profile_picture_filename, session['user_id']))
        else:
            cursor.execute('''UPDATE users SET first_name=%s, last_name=%s, email=%s, phone=%s, profile_picture=%s WHERE id=%s''', (first_name, last_name, email, phone, profile_picture_filename, session['user_id']))

        mysql.connection.commit()

        # Update session
        session['first_name'] = first_name
        session['last_name'] = last_name
        session['full_name'] = f"{first_name} {last_name}".strip()
        session['email'] = email
        session['phone'] = phone
        session['profile_picture'] = profile_picture_filename

        flash('Profile updated successfully!', 'success')

    # Refresh user data from DB (in case session missed something)
    cursor.execute('SELECT * FROM users WHERE id=%s', (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()

    # Ensure session has all data
    session['phone'] = user['phone']
    session['profile_picture'] = user['profile_picture']

    return render_template('profile.html', active_page='profile')

@app.route('/api/remove-profile-picture', methods=['POST'])
def remove_profile_picture():
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get current profile picture
        cursor.execute('SELECT profile_picture FROM users WHERE id=%s', (session['user_id'],))
        user = cursor.fetchone()
        
        if user and user['profile_picture']:
            # Delete file from filesystem
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], user['profile_picture'])
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Update database
            cursor.execute('UPDATE users SET profile_picture=NULL WHERE id=%s', (session['user_id'],))
            mysql.connection.commit()
            
            # Update session
            session['profile_picture'] = None
        
        cursor.close()
        return jsonify({'success': True, 'message': 'Profile picture removed successfully'}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/user', methods=['GET'])
def get_user():
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    return jsonify({
        'success': True,
        'user': {
            'id': session['user_id'],
            'first_name': session['first_name'],
            'last_name': session['last_name'],
            'email': session['email'],
            'full_name': session['full_name'],
            'phone': session['phone'],
            'profile_picture': session.get('profile_picture')
        }
    }), 200

# ---------- LOST ----------
@app.route('/lost', methods=['GET', 'POST'])
def lost():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        if 'user_id' not in session:
            flash('Please log in first.', 'danger')
            return redirect(url_for('login_page'))

        title = request.form.get('title')
        category = request.form.get('category')
        description = request.form.get('description')
        location = request.form.get('location_reported')
        reward = request.form.get('reward') or 0.00
        date_reported = datetime.now().date()

        # insert item
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, reward, status) VALUES (%s, %s, %s, %s, 'lost', %s, %s, %s, %s)''', (session['user_id'], title, description, category, date_reported, location, reward, "Not Yet Found"))

        item_id = cursor.lastrowid
        mysql.connection.commit()
        
        # Update dashboard stats
        update_dashboard_stats(session['user_id'], cursor, mysql)

        features = None

        # insert image + extract AI features
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = f"item_{item_id}_{uuid.uuid4().hex[:8]}.jpg"
                filepath = os.path.join("static/uploads/items", filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)

                # extract features
                features = extract_features(filepath)

                # insert into images table with features
                cursor.execute('''INSERT INTO images (item_id, file_path, ai_features) VALUES (%s, %s, %s)''', (item_id, filepath, str(features.tolist())))
                mysql.connection.commit()

        # prepare details for text matching
        new_details = {
            "title": title,
            "description": description,
            "category": category,
            "location": location
        }

        # auto match
        if features is not None:
            auto_match(item_id, 'lost', features, new_details, cursor, mysql, mail)

        flash('Lost item reported successfully!', 'success')
        cursor.close()
        return redirect(url_for('lost'))

    # GET → show items with reporter info
    cursor.execute("""SELECT i.*, img.file_path AS image_path, u.first_name, u.last_name, u.profile_picture FROM items i LEFT JOIN images img ON i.id = img.item_id LEFT JOIN users u ON i.user_id = u.id WHERE i.type='lost' ORDER BY i.created_at DESC""")
    lost_items = cursor.fetchall()
    cursor.close()
    return render_template('lost.html', items=lost_items, active_page='lost')


@app.route('/found', methods=['GET', 'POST'])
def found():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    if request.method == 'POST':
        if 'user_id' not in session:
            flash('Please log in first.', 'danger')
            return redirect(url_for('login_page'))

        title = request.form.get('title')
        category = request.form.get('category')
        description = request.form.get('description')
        location = request.form.get('location_reported')
        date_reported = datetime.now().date()

        # insert item
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, status) VALUES (%s, %s, %s, %s, 'found', %s, %s, %s)''', (session['user_id'], title, description, category, date_reported, location, "Unclaimed"))

        item_id = cursor.lastrowid
        mysql.connection.commit()
        
        # Update dashboard stats
        update_dashboard_stats(session['user_id'], cursor, mysql)

        features = None

        # insert image + extract AI features
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = f"item_{item_id}_{uuid.uuid4().hex[:8]}.jpg"
                filepath = os.path.join("static/uploads/items", filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                file.save(filepath)

                # extract features
                features = extract_features(filepath)

                # insert into images table with features
                cursor.execute('''INSERT INTO images (item_id, file_path, ai_features) VALUES (%s, %s, %s)''', (item_id, filepath, str(features.tolist())))
                mysql.connection.commit()

        # prepare details for text matching
        new_details = {
            "title": title,
            "description": description,
            "category": category,
            "location": location
        }

        # auto match
        if features is not None:
            auto_match(item_id, 'found', features, new_details, cursor, mysql, mail)

        flash('Found item reported successfully!', 'success')
        cursor.close()
        return redirect(url_for('found'))

    # GET → show items with reporter info
    cursor.execute("""SELECT i.*, img.file_path AS image_path, u.first_name, u.last_name, u.profile_picture FROM items i LEFT JOIN images img ON i.id = img.item_id LEFT JOIN users u ON i.user_id = u.id WHERE i.type='found' ORDER BY i.created_at DESC""")
    found_items = cursor.fetchall()
    cursor.close()

    return render_template('found.html', items=found_items, active_page='found')

# ---------- MATCH ----------
@app.route('/match')
def match():
    if 'user_id' not in session:
        flash("Please log in to view matches.", "danger")
        return redirect(url_for("login_page"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT m.id, m.match_score, m.match_at, m.status, lost.id AS lost_id, lost.title AS lost_title, lost.description AS lost_description, lost.date_reported AS lost_date, u1.first_name AS lost_first_name, u1.last_name AS lost_last_name, u1.profile_picture AS lost_profile, lost_img.file_path AS lost_image, found.id AS found_id, found.title AS found_title, found.description AS found_description, found.date_reported AS found_date, u2.first_name AS found_first_name, u2.last_name AS found_last_name, u2.profile_picture AS found_profile, found_img.file_path AS found_image FROM ai_matches m JOIN items lost ON m.lost_item_id = lost.id JOIN users u1 ON lost.user_id = u1.id LEFT JOIN images lost_img ON lost.id = lost_img.item_id JOIN items found ON m.found_item_id = found.id JOIN users u2 ON found.user_id = u2.id LEFT JOIN images found_img ON found.id = found_img.item_id WHERE lost.user_id = %s OR found.user_id = %s ORDER BY m.match_at DESC""", (session['user_id'], session['user_id']))
    
    matches = cursor.fetchall()
    cursor.close()

    # Normalize image paths for Flask
    for m in matches:
        if m['lost_image']:
            m['lost_image'] = m['lost_image'].replace("\\", "/")  # Windows → web
            if m['lost_image'].startswith("static/"):
                m['lost_image'] = m['lost_image'][7:]  # remove "static/"
        if m['found_image']:
            m['found_image'] = m['found_image'].replace("\\", "/")
            if m['found_image'].startswith("static/"):
                m['found_image'] = m['found_image'][7:]

    return render_template("match.html", matches=matches, active_page="match")

@app.route('/api/match-details/<int:match_id>')
def get_match_details(match_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""SELECT m.id, m.lost_item_id, m.found_item_id, lost.user_id as lost_user_id, lost.title as lost_title, found.user_id as found_user_id, found.title as found_title FROM ai_matches m JOIN items lost ON m.lost_item_id = lost.id JOIN items found ON m.found_item_id = found.id WHERE m.id = %s AND (lost.user_id = %s OR found.user_id = %s)""", (match_id, session['user_id'], session['user_id']))
        
        match_data = cursor.fetchone()
        cursor.close()
        
        if not match_data:
            return jsonify({'success': False, 'errors': ['Match not found or not authorized']}), 404
        
        return jsonify({
            'success': True, 
            'match': {
                'id': match_data['id'],
                'lost_item_id': match_data['lost_item_id'],
                'found_item_id': match_data['found_item_id'],
                'lost_user_id': match_data['lost_user_id'],
                'found_user_id': match_data['found_user_id'],
                'lost_title': match_data['lost_title'],
                'found_title': match_data['found_title']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
