from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash, send_from_directory
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail  
from resnet_model import extract_features, auto_match

import MySQLdb.cursors
import re
import cv2
import json
import numpy as np
from datetime import datetime
import os
import uuid
from PIL import Image
import random
import string
from flask_mail import Message

app = Flask(__name__)
app.secret_key = '071322'  # Change this to a random secret key

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
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'dhhp qhea vvbk zofx')
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

#human detection

def contains_human_faces(image_path):
    """
    Simple face detection: reject only obvious personal photos with large faces
    Allow IDs and documents with small faces
    """
    try:
        print(f"🖼️ Checking image: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        height, width = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            return False
        
        # Detect all faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"👤 Faces found: {len(faces)}")
        
        if len(faces) == 0:
            print("✅ No faces - allowing")
            return False
        
        # Calculate total face area
        total_face_area = sum(w * h for (x, y, w, h) in faces)
        face_area_percentage = (total_face_area / (width * height)) * 100
        
        print(f"📊 Face area: {face_area_percentage:.2f}%")
        
        # CRITICAL: Only reject if faces take up significant space
        # IDs: faces are small (<4% of image)
        # Personal photos: faces are large (>6% of image)
        REJECT_THRESHOLD = 6.0  # Adjust this value as needed
        
        if face_area_percentage > REJECT_THRESHOLD:
            print(f"❌ REJECT: Large faces ({face_area_percentage:.2f}%)")
            return True
        else:
            print(f"✅ ALLOW: Small faces ({face_area_percentage:.2f}%) - likely ID")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def validate_image_file(file):
    """Validate both uploaded files and captured images"""
    if not file or file.filename == '':
        return False, "No image provided"
    
    # Check file size (max 5MB for captured images)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset seek position
    
    if file_size > 5 * 1024 * 1024:  # 5MB
        return False, "Image too large (max 5MB)"
    
    # Check file extension
    if not allowed_file(file.filename):
        return False, "Invalid file type. Please upload PNG, JPG, JPEG, or GIF images."
    
    return True, "Valid"






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


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Update and get dashboard stats
    stats = update_dashboard_stats(session['user_id'], cursor, mysql)
    
    # Get recent activity (last 6 items posted by user) with ALL necessary data
    cursor.execute("""SELECT 
        i.id, i.title, i.type, i.category, i.description, i.location_reported, 
        i.status, i.created_at, i.date_reported,
        img.file_path as image_path,
        u.first_name, u.last_name, u.profile_picture
    FROM items i 
    LEFT JOIN images img ON i.id = img.item_id 
    LEFT JOIN users u ON i.user_id = u.id 
    WHERE i.user_id = %s 
    ORDER BY i.created_at DESC LIMIT 6""", (session['user_id'],))
    
    recent_items = cursor.fetchall()
    
    # Fix image paths for display and add missing data attributes
    for item in recent_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            if item['image_path'].startswith("static/"):
                item['image_path'] = item['image_path'][7:]
        
        # Ensure all required fields have values
        item['description'] = item['description'] or 'No description provided'
        item['category'] = item['category'] or 'Not specified'
        item['status'] = item['status'] or 'active'
        item['date_reported'] = item['date_reported'] or item['created_at']
        
        # Format date for modal display
        if item['date_reported']:
            if isinstance(item['date_reported'], str):
                # If it's a string, try to parse it
                try:
                    date_obj = datetime.strptime(item['date_reported'], '%Y-%m-%d %H:%M:%S')
                    item['formatted_date'] = date_obj.strftime('%B %d, %Y at %I:%M %p')
                except:
                    item['formatted_date'] = 'Date not available'
            else:
                # If it's a datetime object
                item['formatted_date'] = item['date_reported'].strftime('%B %d, %Y at %I:%M %p')
        else:
            item['formatted_date'] = 'Date not available'
    
    cursor.execute("""SELECT f.message, f.rating, f.created_at, u.first_name, u.last_name, u.profile_picture FROM feedback f JOIN users u ON f.user_id = u.id WHERE f.is_public = 1 ORDER BY f.created_at DESC LIMIT 6""")
    recent_feedback = cursor.fetchall()
    
    cursor.close()
    
    # Get stored form data for repopulating modals
    lost_form_data = session.pop('lost_form_data', None) if 'lost_form_data' in session else None
    found_form_data = session.pop('found_form_data', None) if 'found_form_data' in session else None
    
    return render_template('dashboard.html', active_page='dashboard', user=session, stats=stats, recent_items=recent_items, recent_feedback=recent_feedback, lost_form_data=lost_form_data, found_form_data=found_form_data)

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

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    # Redirect to new OTP flow
    return jsonify({
        'success': False, 
        'errors': ['Please use the new registration process with email verification.'],
        'redirect': '/api/send-otp'
    }), 400

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
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
        cursor.close()
        
        if existing_user:
            return jsonify({'success': False, 'errors': ['Email already registered']}), 400
        
        # Generate and store OTP in session
        otp = generate_otp()
        session['registration_otp'] = otp
        session['registration_data'] = {
            'firstName': first_name,
            'lastName': last_name,
            'email': email,
            'phone': phone,
            'password': password
        }
        session['otp_timestamp'] = datetime.now().timestamp()
        
        # Send OTP email
        if send_otp_email(email, otp, first_name):
            return jsonify({
                'success': True, 
                'message': 'Verification code sent to your email!'
            }), 200
        else:
            return jsonify({'success': False, 'errors': ['Failed to send verification email. Please try again.']}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        entered_otp = data.get('otpCode', '').strip()
        
        if not entered_otp:
            return jsonify({'success': False, 'errors': ['Please enter the verification code']}), 400
        
        # Check if OTP exists in session
        if 'registration_otp' not in session or 'registration_data' not in session:
            return jsonify({'success': False, 'errors': ['Verification session expired. Please try registering again.']}), 400
        
        # Check OTP expiry (10 minutes)
        otp_timestamp = session.get('otp_timestamp', 0)
        current_timestamp = datetime.now().timestamp()
        if current_timestamp - otp_timestamp > 600:  # 10 minutes
            # Clear session data
            session.pop('registration_otp', None)
            session.pop('registration_data', None)
            session.pop('otp_timestamp', None)
            return jsonify({'success': False, 'errors': ['Verification code expired. Please try registering again.']}), 400
        
        # Verify OTP
        stored_otp = session.get('registration_otp')
        if entered_otp != stored_otp:
            return jsonify({'success': False, 'errors': ['Invalid verification code. Please try again.']}), 400
        
        # OTP is valid, create the account
        registration_data = session.get('registration_data')
        
        # Check if user already exists (double-check)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (registration_data['email'],))
        existing_user = cursor.fetchone()
        
        if existing_user:
            cursor.close()
            return jsonify({'success': False, 'errors': ['Email already registered']}), 400
        
        # Hash password and create user
        hashed_password = generate_password_hash(registration_data['password'])
        
        cursor.execute('''INSERT INTO users (first_name, last_name, email, phone, password) VALUES (%s, %s, %s, %s, %s)''', 
                      (registration_data['firstName'], registration_data['lastName'], 
                       registration_data['email'], registration_data['phone'], hashed_password))
        
        mysql.connection.commit()
        cursor.close()
        
        # Clear session data
        session.pop('registration_otp', None)
        session.pop('registration_data', None)
        session.pop('otp_timestamp', None)
        
        return jsonify({
            'success': True, 
            'message': 'Account created successfully! You can now log in.',
            'redirect': '/login'
            
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
        action_type = data.get('action_type')
        
        print(f"DEBUG: Claim request - notification: {notification_id}, item: {item_id}, action: {action_type}")
        
        if not all([notification_id, item_id, action_type]):
            return jsonify({'success': False, 'errors': ['Missing required fields']}), 400
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get the item being acted upon
        cursor.execute("SELECT id, user_id, type, status FROM items WHERE id = %s", (item_id,))
        target_item = cursor.fetchone()
        
        if not target_item:
            cursor.close()
            return jsonify({'success': False, 'errors': ['Item not found']}), 404
        
        print(f"DEBUG: Target item - id: {target_item['id']}, type: {target_item['type']}, status: {target_item['status']}, user_id: {target_item['user_id']}")
        
        # Get match details to verify authorization
        cursor.execute("""SELECT n.*, m.id as match_id, m.lost_item_id, m.found_item_id, 
                         lost.user_id as lost_user_id, lost.status as lost_status,
                         found.user_id as found_user_id, found.status as found_status 
                         FROM notifications n 
                         JOIN ai_matches m ON n.match_id = m.id 
                         JOIN items lost ON m.lost_item_id = lost.id 
                         JOIN items found ON m.found_item_id = found.id 
                         WHERE n.id = %s""", (notification_id,))
        
        match_data = cursor.fetchone()
        if not match_data:
            cursor.close()
            return jsonify({'success': False, 'errors': ['Match not found']}), 404
        
        user_id = session['user_id']
        current_time = datetime.now()
        
        print(f"DEBUG: Match details - lost_item: {match_data['lost_item_id']} (user: {match_data['lost_user_id']}), found_item: {match_data['found_item_id']} (user: {match_data['found_user_id']})")
        
        # CORRECTED LOGIC - REVERSED:
        valid_action = False
        new_status = None
        update_query = None
        update_params = None
        
        # Case 1: Lost item owner clicking "Mark as Claimed" (they are claiming they got it back)
        if (action_type == 'claimed' and 
            user_id == match_data['lost_user_id'] and 
            target_item['id'] == match_data['lost_item_id'] and
            target_item['type'] == 'lost'):
            
            valid_action = True
            new_status = 'claimed'  # Lost item gets CLAIMED back by owner
            update_query = "UPDATE items SET status = %s, date_returned = %s WHERE id = %s"
            update_params = (new_status, current_time, item_id)
            print(f"DEBUG: LOST item owner claiming item back - setting to 'claimed'")
            
        # Case 2: Found item owner clicking "Mark as Returned" (they are returning it to owner)  
        elif (action_type == 'returned' and 
              user_id == match_data['found_user_id'] and 
              target_item['id'] == match_data['found_item_id'] and
              target_item['type'] == 'found'):
            
            valid_action = True
            new_status = 'returned'  # Found item gets RETURNED to owner
            update_query = "UPDATE items SET status = %s, claimed_by = %s, date_claimed = %s WHERE id = %s"
            update_params = (new_status, match_data['lost_user_id'], current_time, item_id)
            print(f"DEBUG: FOUND item owner returning item - setting to 'returned'")
        
        if not valid_action:
            cursor.close()
            error_msg = f"Invalid action: User {user_id} cannot perform {action_type} on item {item_id}"
            print(f"DEBUG: {error_msg}")
            return jsonify({'success': False, 'errors': [error_msg]}), 403
        
        # Check if item is already in the target status
        if target_item['status'] == new_status:
            cursor.close()
            error_msg = f"Item is already marked as {new_status}"
            print(f"DEBUG: {error_msg}")
            return jsonify({'success': False, 'errors': [error_msg]}), 400
        
        # Execute the update
        cursor.execute(update_query, update_params)
        
        # Mark notification as read
        cursor.execute("UPDATE notifications SET is_read = 1 WHERE id = %s", (notification_id,))
        mysql.connection.commit()
        
        print(f"DEBUG: Successfully updated item {item_id} from '{target_item['status']}' to '{new_status}'")
        
        # Check if both actions are completed - ALSO REVERSED
        cursor.execute("""SELECT lost.status as lost_status, found.status as found_status 
                         FROM ai_matches m 
                         JOIN items lost ON m.lost_item_id = lost.id 
                         JOIN items found ON m.found_item_id = found.id 
                         WHERE m.id = %s""", (match_data['match_id'],))
        
        status_check = cursor.fetchone()
        both_completed = (status_check['lost_status'] == 'claimed' and status_check['found_status'] == 'returned')
        
        print(f"DEBUG: Status check - Lost: {status_check['lost_status']}, Found: {status_check['found_status']}, Complete: {both_completed}")
        
        if both_completed:
            # Create claim record
            cursor.execute("""INSERT INTO claims (lost_user_id, found_user_id, item_lost_id, item_found_id, claim_date) 
                            VALUES (%s, %s, %s, %s, NOW())""", 
                          (match_data['lost_user_id'], match_data['found_user_id'], 
                           match_data['lost_item_id'], match_data['found_item_id']))
            
            # Mark match as resolved
            cursor.execute("UPDATE ai_matches SET status = 'resolved' WHERE id = %s", (match_data['match_id'],))
            
            mysql.connection.commit()
            print("DEBUG: Match resolved and claim recorded!")
        
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': f'Success! Item status updated to {new_status}.',
            'both_completed': both_completed,
            'action_type': action_type
        }), 200
        
    except Exception as e:
        print(f"ERROR in claim_item: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/claim-status/<int:notification_id>')
def get_claim_status(notification_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get notification details
        cursor.execute("""SELECT match_id FROM notifications WHERE id = %s""", (notification_id,))
        notification = cursor.fetchone()
        if not notification:
            return jsonify({'success': False, 'errors': ['Notification not found']}), 404
        
        match_id = notification['match_id']
        
        # Get match and item status details
        cursor.execute("""SELECT 
            m.status as match_status, 
            lost.id as lost_item_id, lost.user_id as lost_user_id, lost.status as lost_item_status,
            found.id as found_item_id, found.user_id as found_user_id, found.status as found_item_status,
            u1.first_name as lost_first_name, u1.last_name as lost_last_name,
            u2.first_name as found_first_name, u2.last_name as found_last_name 
            FROM ai_matches m 
            JOIN items lost ON m.lost_item_id = lost.id 
            JOIN users u1 ON lost.user_id = u1.id 
            JOIN items found ON m.found_item_id = found.id 
            JOIN users u2 ON found.user_id = u2.id 
            WHERE m.id = %s""", (match_id,))
        
        match_data = cursor.fetchone()
        cursor.close()
        
        if not match_data:
            return jsonify({'success': False, 'errors': ['Match not found']}), 404
        
        claim_status = {
            'claimed': None,
            'returned': None,
            'both_completed': False
        }
        
        # REVERSED: Check if lost item is claimed (by the lost item owner)
        if match_data['lost_item_status'] == 'claimed':
            claim_status['claimed'] = {
                'user_id': match_data['lost_user_id'],
                'name': f"{match_data['lost_first_name']} {match_data['lost_last_name']}",
                'status': 'completed',
                'item_id': match_data['lost_item_id']
            }
        
        # REVERSED: Check if found item is returned (by the found item owner)
        if match_data['found_item_status'] == 'returned':
            claim_status['returned'] = {
                'user_id': match_data['found_user_id'],
                'name': f"{match_data['found_first_name']} {match_data['found_last_name']}",
                'status': 'completed',
                'item_id': match_data['found_item_id']
            }
        
        claim_status['both_completed'] = (match_data['lost_item_status'] == 'claimed' and 
                                         match_data['found_item_status'] == 'returned')
        
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
        reward = request.form.get('reward', '').strip()
        date_reported = datetime.now()

        # Validate required fields
        if not all([title, category, description, location]):
            flash('Please fill in all required fields.', 'danger')
            # Store form data for repopulation
            form_data = {
                'title': title,
                'category': category,
                'description': description,
                'location_reported': location,
                'reward': reward
            }
            session['lost_form_data'] = form_data
            return redirect(url_for('dashboard') + '#lostItemModal')

        # insert item
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, reward, status) VALUES (%s, %s, %s, %s, 'lost', %s, %s, %s, %s)''', (session['user_id'], title, description, category, date_reported, location, reward, "Not Yet Found"))

        item_id = cursor.lastrowid
        mysql.connection.commit()
        
        # Update dashboard stats
        update_dashboard_stats(session['user_id'], cursor, mysql)

        features = None

        # insert image + extract AI features WITH FACE DETECTION
        if 'image' in request.files:
            file = request.files['image']
            
            # Validate image file
            if file and file.filename != '':
                is_valid, message = validate_image_file(file)
                if not is_valid:
                    # Delete the item record since validation failed
                    cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
                    mysql.connection.commit()
                    
                    # Store form data for repopulation
                    form_data = {
                        'title': title,
                        'category': category,
                        'description': description,
                        'location_reported': location,
                        'reward': reward
                    }
                    flash(f'❌ {message} LOST_FORM_DATA:{json.dumps(form_data)}', 'danger')
                    return redirect(url_for('dashboard') + '#lostItemModal')
                
                if allowed_file(file.filename):
                    print(f"🔍 Processing image upload for item {item_id}")
                    
                    # Create temp directory if it doesn't exist
                    temp_dir = "static/uploads/temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Save file temporarily for face detection
                    temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    file.save(temp_filepath)
                    print(f"📁 Saved temporary file: {temp_filepath}")
                    
                    # Check for human faces
                    print("🔍 Running face detection...")
                    has_faces = contains_human_faces(temp_filepath)
                    print(f"🎭 Face detection result: {has_faces}")
                    
                    if has_faces:
                        # Delete temp file and return error
                        os.remove(temp_filepath)
                        # Also delete the item record since we're rejecting the submission
                        cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
                        mysql.connection.commit()
                        print("❌ Face detected - rejecting submission")
                        
                        # Store form data in flash message as JSON
                        form_data = {
                            'title': title,
                            'category': category,
                            'description': description,
                            'location_reported': location,
                            'reward': reward
                        }
                        flash(f'❌ Image contains human faces. Please upload only images of the item itself for privacy and security reasons. LOST_FORM_DATA:{json.dumps(form_data)}', 'danger')
                        # Redirect back to dashboard with modal open
                        return redirect(url_for('dashboard') + '#lostItemModal')
                    
                    # If no faces, proceed with original processing
                    print("✅ No faces detected - proceeding with item creation")
                    filename = f"item_{item_id}_{uuid.uuid4().hex[:8]}.jpg"
                    filepath = os.path.join("static/uploads/items", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    os.rename(temp_filepath, filepath)  # Move from temp to final location

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
    
    # Get the item_id from URL parameter if present
    show_item_id = request.args.get('show_item')
    
    # Fix image paths for display
    for item in lost_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            if item['image_path'].startswith("static/"):
                item['image_path'] = item['image_path'][7:]
    
    cursor.close()
    return render_template('lost.html', items=lost_items, active_page='lost', show_item_id=show_item_id)

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
        date_reported = datetime.now()

        # Validate required fields
        if not all([title, category, description, location]):
            flash('Please fill in all required fields.', 'danger')
            # Store form data for repopulation
            form_data = {
                'title': title,
                'category': category,
                'description': description,
                'location_reported': location
            }
            session['found_form_data'] = form_data
            return redirect(url_for('dashboard') + '#foundItemModal')

        # insert item
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, status) VALUES (%s, %s, %s, %s, 'found', %s, %s, %s)''', (session['user_id'], title, description, category, date_reported, location, "Unclaimed"))

        item_id = cursor.lastrowid
        mysql.connection.commit()
        
        # Update dashboard stats
        update_dashboard_stats(session['user_id'], cursor, mysql)

        features = None

        # insert image + extract AI features WITH FACE DETECTION
        if 'image' in request.files:
            file = request.files['image']
            
            # Validate image file
            if file and file.filename != '':
                is_valid, message = validate_image_file(file)
                if not is_valid:
                    # Delete the item record since validation failed
                    cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
                    mysql.connection.commit()
                    
                    # Store form data for repopulation
                    form_data = {
                        'title': title,
                        'category': category,
                        'description': description,
                        'location_reported': location
                    }
                    flash(f'❌ {message} FOUND_FORM_DATA:{json.dumps(form_data)}', 'danger')
                    return redirect(url_for('dashboard') + '#foundItemModal')
                
                if allowed_file(file.filename):
                    print(f"🔍 Processing image upload for found item {item_id}")
                    
                    # Create temp directory if it doesn't exist
                    temp_dir = "static/uploads/temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Save file temporarily for face detection
                    temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    file.save(temp_filepath)
                    print(f"📁 Saved temporary file: {temp_filepath}")
                    
                    # Check for human faces
                    print("🔍 Running face detection...")
                    has_faces = contains_human_faces(temp_filepath)
                    print(f"🎭 Face detection result: {has_faces}")
                    
                    if has_faces:
                        # Delete temp file and return error
                        os.remove(temp_filepath)
                        # Also delete the item record since we're rejecting the submission
                        cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
                        mysql.connection.commit()
                        print("❌ Face detected - rejecting submission")
                        
                        # Store form data in flash message as JSON
                        form_data = {
                            'title': title,
                            'category': category,
                            'description': description,
                            'location_reported': location
                        }
                        flash(f'❌ Image contains human faces. Please upload only images of the item itself for privacy and security reasons. FOUND_FORM_DATA:{json.dumps(form_data)}', 'danger')
                        # Redirect back to dashboard with modal open
                        return redirect(url_for('dashboard') + '#foundItemModal')
                    
                    # If no faces, proceed with original processing
                    print("✅ No faces detected - proceeding with item creation")
                    filename = f"item_{item_id}_{uuid.uuid4().hex[:8]}.jpg"
                    filepath = os.path.join("static/uploads/items", filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    os.rename(temp_filepath, filepath)  # Move from temp to final location

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
    
    # Get the item_id from URL parameter if present
    show_item_id = request.args.get('show_item')
    
    # Fix image paths for display
    for item in found_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            if item['image_path'].startswith("static/"):
                item['image_path'] = item['image_path'][7:]
    
    cursor.close()
    return render_template('found.html', items=found_items, active_page='found', show_item_id=show_item_id)

@app.route('/posted')
def posted():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Get all items posted by the user with image paths
    cursor.execute("""SELECT 
        i.*, 
        img.file_path as image_path,
        COUNT(DISTINCT m.id) as match_count
    FROM items i 
    LEFT JOIN images img ON i.id = img.item_id 
    LEFT JOIN ai_matches m ON (m.lost_item_id = i.id OR m.found_item_id = i.id)
    WHERE i.user_id = %s 
    GROUP BY i.id
    ORDER BY i.created_at DESC""", (session['user_id'],))
    
    posted_items = cursor.fetchall()
    
    # Fix image paths for display
    for item in posted_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            if item['image_path'].startswith("static/"):
                item['image_path'] = item['image_path'][7:]
    
    cursor.close()
    
    return render_template('posted.html', 
                         posted_items=posted_items, 
                         active_page='posted')

@app.route('/api/get-item/<int:item_id>')
def get_item(item_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Verify ownership and get complete item details
        cursor.execute("""SELECT * FROM items WHERE id = %s AND user_id = %s""", 
                      (item_id, session['user_id']))
        item = cursor.fetchone()
        cursor.close()
        
        if not item:
            return jsonify({'success': False, 'errors': ['Item not found or not authorized']}), 404
        
        return jsonify({
            'success': True, 
            'item': {
                'id': item['id'],
                'title': item['title'],
                'description': item['description'],
                'category': item['category'],
                'type': item['type'],
                'location_reported': item['location_reported'],
                'reward': item['reward'],
                'status': item['status'],
                'date_reported': item['date_reported'].strftime('%Y-%m-%d') if item['date_reported'] else None,
                'created_at': item['created_at'].strftime('%Y-%m-%d %H:%M:%S') if item['created_at'] else None
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/delete-item/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Verify ownership
        cursor.execute("SELECT user_id, type FROM items WHERE id = %s", (item_id,))
        item = cursor.fetchone()
        
        if not item:
            return jsonify({'success': False, 'errors': ['Item not found']}), 404
        
        if item['user_id'] != session['user_id']:
            return jsonify({'success': False, 'errors': ['Not authorized']}), 403
        
        # Get image path to delete file
        cursor.execute("SELECT file_path FROM images WHERE item_id = %s", (item_id,))
        image = cursor.fetchone()
        
        # Delete from database
        cursor.execute("DELETE FROM images WHERE item_id = %s", (item_id,))
        cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
        
        mysql.connection.commit()
        cursor.close()
        
        # Delete image file if exists
        if image and image['file_path']:
            try:
                file_path = image['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting image file: {e}")
        
        return jsonify({'success': True, 'message': 'Item deleted successfully'}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/update-item/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        data = request.get_json()
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Verify ownership
        cursor.execute("SELECT user_id, type FROM items WHERE id = %s", (item_id,))
        item = cursor.fetchone()
        
        if not item or item['user_id'] != session['user_id']:
            return jsonify({'success': False, 'errors': ['Not authorized']}), 403
        
        # Update item with all fields
        cursor.execute("""UPDATE items SET 
            title = %s, description = %s, category = %s, 
            location_reported = %s, reward = %s 
            WHERE id = %s""", 
            (data.get('title'), data.get('description'), data.get('category'),
             data.get('location_reported'), data.get('reward'), item_id))
        
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({'success': True, 'message': 'Item updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500
    
    

# ---------- MATCH ----------
@app.route('/match')
def match():
    if 'user_id' not in session:
        flash("Please log in to view matches.", "danger")
        return redirect(url_for("login_page"))

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT m.id, m.match_score, m.match_at, m.status, lost.id AS lost_id, lost.title AS lost_title, lost.description AS lost_description, lost.category AS lost_category, lost.date_reported AS lost_date_reported, u1.first_name AS lost_first_name, u1.last_name AS lost_last_name, u1.profile_picture AS lost_profile, lost_img.file_path AS lost_image, found.id AS found_id, found.title AS found_title, found.description AS found_description, found.category AS found_category, found.date_reported AS found_date_reported, u2.first_name AS found_first_name, u2.last_name AS found_last_name, u2.profile_picture AS found_profile, found_img.file_path AS found_image FROM ai_matches m JOIN items lost ON m.lost_item_id = lost.id JOIN users u1 ON lost.user_id = u1.id LEFT JOIN images lost_img ON lost.id = lost_img.item_id JOIN items found ON m.found_item_id = found.id JOIN users u2 ON found.user_id = u2.id LEFT JOIN images found_img ON found.id = found_img.item_id WHERE lost.user_id = %s OR found.user_id = %s ORDER BY m.match_at DESC""", (session['user_id'], session['user_id']))
    
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

def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, otp, first_name):
    """Send OTP via email"""
    try:
        msg = Message(
            subject='Verify Your Email - Reunited',
            recipients=[email],
            html=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #1E3A8A; margin-bottom: 10px;">🤝 Reunited</h1>
                    <h2 style="color: #374151; font-weight: normal;">Email Verification</h2>
                </div>
                
                <div style="background: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">
                        Hi {first_name},
                    </p>
                    <p style="font-size: 16px; color: #374151; margin-bottom: 30px;">
                        Welcome to Reunited! Please use the verification code below to complete your registration:
                    </p>
                    
                    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h1 style="font-size: 36px; color: #1E3A8A; letter-spacing: 8px; margin: 0; font-family: monospace;">
                            {otp}
                        </h1>
                    </div>
                    
                    <p style="font-size: 14px; color: #6b7280; margin-top: 20px;">
                        This code will expire in 10 minutes. If you didn't request this, please ignore this email.
                    </p>
                </div>
                
                <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                    <p style="font-size: 12px; color: #9ca3af;">
                        © 2025 Reunited. Helping reconnect lost items with their owners.
                    </p>
                </div>
            </div>
            '''
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending OTP email: {e}")
        return False

# Serve uploaded files
@app.route('/uploads/profile_pictures/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve uploaded item images
@app.route('/uploads/items/<filename>')
def uploaded_item_file(filename):
    return send_from_directory('static/uploads/items', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
