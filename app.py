from flask import Flask, request, jsonify, session, render_template, redirect, url_for, flash, send_from_directory
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail  
from resnet_model import extract_features, auto_match
from datetime import datetime, timedelta  # Add timedelta here

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
import requests
import base64
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', '071322')  # Use env var

# MySQL Configuration (Railway + local fallback)
app.config['MYSQL_HOST'] = os.getenv('MYSQLHOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQLUSER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQLPASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQLDATABASE', 'reunited_db')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQLPORT', 3306))

# Add SSL configuration for Railway MySQL
app.config['MYSQL_SSL_MODE'] = 'REQUIRED'
app.config['MYSQL_SSL_CA'] = None  # Railway handles SSL automatically

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

# PayMongo Configuration
PAYMONGO_SECRET_KEY = os.getenv('PAYMONGO_SECRET_KEY')
PAYMONGO_PUBLIC_KEY = os.getenv('PAYMONGO_PUBLIC_KEY')

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

#human detection

#human detection

def contains_human_faces(image_path):
    """
    Smart face detection that ALLOWS ID cards but REJECTS personal photos.
    IDs are identified by their rectangular shape, text, and small face proportions.
    """
    try:
        print(f"🖼️ Checking image: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        height, width = img.shape[:2]
        print(f"📏 Image dimensions: {width}x{height}")
        
        # Convert to grayscale for multiple analyses
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if face_cascade.empty() or profile_cascade.empty():
            print("⚠️ Cascade classifiers not loaded properly")
            return False
        
        # ---------------------------------------------------------
        # STEP 1: Check if this looks like an ID CARD (ALLOW if yes)
        # ---------------------------------------------------------
        
        # 1A. Check for rectangular document-like shape (ID cards are usually rectangular)
        # Convert to binary for contour detection
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for large, rectangular contours (potential ID card)
        id_card_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.1):  # At least 10% of image area
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # ID cards typically have aspect ratios between 1.4 and 1.8 (credit card: 1.586)
                    if 1.3 <= aspect_ratio <= 1.9:
                        id_card_detected = True
                        print(f"📇 ID Card detected: {w}x{h} (aspect: {aspect_ratio:.2f})")
                        
                        # Additional check: ID cards often have text/edges around the border
                        # Extract the potential ID region
                        id_region = img[y:y+h, x:x+w]
                        
                        # Check for text-like features (high edge density)
                        edges = cv2.Canny(id_region, 50, 150)
                        edge_density = np.sum(edges > 0) / (w * h) * 100
                        
                        if edge_density > 5:  # IDs have lots of text/edges
                            print(f"   📝 High edge density ({edge_density:.1f}%) - looks like document")
                            return False  # ALLOW ID cards immediately
        
        # 1B. Check for ID-like features: Small face in a structured document
        # Detect ALL faces in the image
        faces_frontal = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Check if this is an ID photo (small face in structured layout)
        for (x, y, w, h) in faces_frontal:
            face_area = w * h
            image_area = width * height
            face_percentage = (face_area / image_area) * 100
            
            print(f"👤 Face detected: {w}x{h} ({face_percentage:.2f}% of image)")
            
            # ID photos typically have faces that are 2-4% of total image area
            if 1.5 <= face_percentage <= 4.5:
                print(f"📇 Face size suggests ID photo ({face_percentage:.2f}%)")
                
                # Check if face is in upper portion (typical for IDs)
                face_center_y = y + h/2
                if face_center_y < height * 0.4:  # Face in top 40% of image
                    print(f"   📍 Face position suggests ID layout")
                    return False  # ALLOW ID cards
        
        # ---------------------------------------------------------
        # STEP 2: Only if NOT an ID, check for personal photos to REJECT
        # ---------------------------------------------------------
        
        # Detect faces with stricter parameters for personal photos
        faces_frontal = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Increased for stricter detection
            minNeighbors=6,   # Increased to reduce false positives
            minSize=(40, 40), # Minimum face size increased
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces_profile = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Combine all detected faces
        all_faces = list(faces_frontal) + list(faces_profile)
        
        if len(all_faces) == 0:
            print("✅ No faces detected - allowing image")
            return False
        
        print(f"👤 Total faces found for personal photo check: {len(all_faces)}")
        
        # Validate detected faces for personal photos
        valid_faces = []
        for i, (x, y, w, h) in enumerate(all_faces):
            # Skip very small detections
            if w < 40 or h < 40:
                continue
                
            face_region = img[y:y+h, x:x+w]
            
            # Check skin tone (personal photos usually show skin)
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Skin tone ranges
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            lower_skin2 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
            
            skin_mask1 = cv2.inRange(hsv_face, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv_face, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = w * h
            skin_percentage = (skin_pixels / total_pixels) * 100
            
            # Check edge density (faces have features)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_face, 100, 200)
            edge_density = np.sum(edges > 0) / total_pixels * 100
            
            # Validate as personal photo face
            if skin_percentage > 15 or edge_density > 10:
                valid_faces.append((x, y, w, h))
                print(f"  Valid face {i+1}: {w}x{h}, Skin: {skin_percentage:.1f}%, Edges: {edge_density:.1f}%")
        
        if len(valid_faces) == 0:
            print("✅ No valid personal photo faces detected - allowing image")
            return False
        
        # Calculate total valid face area for personal photos
        total_face_area = sum(w * h for (x, y, w, h) in valid_faces)
        face_area_percentage = (total_face_area / (width * height)) * 100
        
        print(f"📊 Personal photo face area: {total_face_area}px ({face_area_percentage:.2f}% of image)")
        
        # FINAL DECISION FOR PERSONAL PHOTOS:
        # Personal photos: faces are large (>6% of image) → REJECT
        # Casual photos with small faces: faces are small (<6%) → ALLOW
        PERSONAL_PHOTO_THRESHOLD = 6.0
        
        if face_area_percentage > PERSONAL_PHOTO_THRESHOLD:
            print(f"❌ REJECT: Large human faces detected ({face_area_percentage:.2f}%) - appears to be personal photo")
            return True
        else:
            print(f"✅ ALLOW: Casual photo with small faces ({face_area_percentage:.2f}%)")
            return False
            
    except Exception as e:
        print(f"❌ Error in face detection: {e}")
        import traceback
        traceback.print_exc()
        return False  # Allow on error to not block legitimate uploads

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


# Admin dashboard route - FIXED
@app.route('/admindashboard')
def admin_dashboard():
    # Check if user is admin
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    
    # ✅ REMOVE the database queries - let the API handle it
    # The frontend will fetch data via JavaScript APIs
    
    return render_template('admindashboard.html', active_page='admin_dashboard')

@app.route('/api/admin/category-monthly')
def admin_category_monthly():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Get period parameter
    period = request.args.get('period', 'all_time')
    
    try:
        # Build conditions list
        conditions = [
            "category IS NOT NULL",
            "created_at IS NOT NULL"
        ]
        
        if period == 'current_month':
            conditions.append("MONTH(created_at) = MONTH(CURRENT_DATE())")
            conditions.append("YEAR(created_at) = YEAR(CURRENT_DATE())")
        elif period == 'last_month':
            conditions.append("MONTH(created_at) = MONTH(CURRENT_DATE() - INTERVAL 1 MONTH)")
            conditions.append("YEAR(created_at) = YEAR(CURRENT_DATE() - INTERVAL 1 MONTH)")
        
        # Build WHERE clause
        where_clause = "WHERE " + " AND ".join(conditions)
        
        print(f"🔍 Period: {period}")
        print(f"🔍 Conditions: {conditions}")
        
        # FIXED QUERY - Proper ordering by actual date, not formatted string
        query = f"""
            SELECT 
                DATE_FORMAT(created_at, '%b %Y') as month_name,
                category,
                COUNT(*) as total_items
            FROM items 
            {where_clause}
            GROUP BY DATE_FORMAT(created_at, '%Y-%m'), category
            ORDER BY DATE_FORMAT(created_at, '%Y-%m') ASC, category
        """
        
        print(f"🔍 Query: {query}")
        cursor.execute(query)
        data = cursor.fetchall()
        
        # Debug: Print all data
        print(f"📊 Raw data from query:")
        for row in data:
            print(f"  - {row['month_name']}: {row['category']} = {row['total_items']}")
        
        # FIXED MONTHS QUERY - Get all distinct months with proper ordering
        month_conditions = conditions.copy()
        if 'category IS NOT NULL' in month_conditions:
            month_conditions.remove('category IS NOT NULL')
        
        month_where = "WHERE " + " AND ".join(month_conditions)
        months_query = f"""
            SELECT DISTINCT DATE_FORMAT(created_at, '%b %Y') as month_name
            FROM items 
            {month_where}
            ORDER BY DATE_FORMAT(created_at, '%Y-%m') ASC
        """
        
        cursor.execute(months_query)
        months = [row['month_name'] for row in cursor.fetchall()]
        
        # Get categories
        categories_query = f"""
            SELECT DISTINCT category
            FROM items 
            {where_clause}
            ORDER BY category
        """
        cursor.execute(categories_query)
        categories = [row['category'] for row in cursor.fetchall()]
        
        # Additional debug query: Check items MATCHING THE CURRENT PERIOD
        debug_query = f"""
            SELECT 
                id,
                category,
                created_at,
                DATE_FORMAT(created_at, '%b %Y') as formatted_date,
                DATE_FORMAT(created_at, '%Y-%m') as sort_date
            FROM items 
            {where_clause}
            ORDER BY created_at
        """
        cursor.execute(debug_query)
        all_items = cursor.fetchall()
        
        print(f"\n📋 ALL ITEMS IN DATABASE:")
        for item in all_items:
            print(f"  ID {item['id']}: {item['category']} - {item['created_at']} ({item['formatted_date']}) [Sort: {item['sort_date']}]")
        
        cursor.close()
        
        # Debug
        print(f"\n📅 Months: {months}")
        print(f"📋 Categories: {categories}")
        print(f"📊 Data rows: {len(data)}")
        print(f"📊 Total items in DB: {len(all_items)}")
        
        return jsonify({
            'success': True,
            'chart_data': {
                'months': months,
                'categories': categories,
                'data': data
            },
            'period': period,
            'debug': {
                'total_items_in_db': len(all_items),
                'items_in_chart': sum(d['total_items'] for d in data)
            }
        })
        
    except Exception as e:
        print(f"❌ Error in category-monthly: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# API route for admin statistics - FIXED
@app.route('/api/admin/stats')
def admin_stats():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Get period parameter (default to 6 months)
    period = request.args.get('period', '6')
    
    # Validate period parameter
    try:
        period_int = int(period)
        if period_int not in [6, 12, 24]:
            period_int = 6
    except ValueError:
        period_int = 6
    
    # Real-time statistics
    cursor.execute("SELECT COUNT(*) as total_users FROM users WHERE is_active = TRUE")
    total_users = cursor.fetchone()['total_users']
    
    cursor.execute("SELECT COUNT(*) as total_items FROM items")
    total_items = cursor.fetchone()['total_items']
    
    cursor.execute("""
        SELECT COUNT(*) as total_matches 
        FROM ai_matches 
        WHERE status = 'resolved'
    """)
    total_matches = cursor.fetchone()['total_matches']
    
    cursor.execute("""
        SELECT COALESCE(SUM(CASE WHEN payment_status = 'paid' THEN 20 ELSE 0 END), 0) as total_revenue
        FROM items 
        WHERE type = 'lost'
    """)
    total_revenue = cursor.fetchone()['total_revenue']
    
    # Get item counts for quick stats
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN type = 'lost' THEN 1 ELSE 0 END) as lost_items,
            SUM(CASE WHEN type = 'found' THEN 1 ELSE 0 END) as found_items,
            SUM(CASE WHEN status IN ('claimed', 'returned') THEN 1 ELSE 0 END) as resolved_items,
            SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) as paid_items
        FROM items
    """)
    item_counts = cursor.fetchone()
    
    # Monthly items data for chart (both lost and found)
    cursor.execute(f"""
        SELECT 
            DATE_FORMAT(created_at, '%b %Y') as month_name,
            COUNT(*) as total_items,
            SUM(CASE WHEN type = 'lost' THEN 1 ELSE 0 END) as lost_items,
            SUM(CASE WHEN type = 'found' THEN 1 ELSE 0 END) as found_items
        FROM items 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL {period_int} MONTH)
        GROUP BY DATE_FORMAT(created_at, '%Y-%m'), month_name
        ORDER BY MIN(created_at) ASC
        LIMIT 12
    """)
    monthly_data = cursor.fetchall()
    
    # Get recent activity for the dashboard
    cursor.execute("""
        (
            SELECT 
                'user_registered' as type,
                CONCAT('New user registered: ', first_name, ' ', last_name) as message,
                created_at as timestamp
            FROM users 
            ORDER BY created_at DESC 
            LIMIT 3
        )
        UNION ALL
        (
            SELECT 
                'item_posted' as type,
                CONCAT('New ', type, ' item: ', title) as message,
                created_at as timestamp
            FROM items 
            ORDER BY created_at DESC 
            LIMIT 3
        )
        UNION ALL
        (
            SELECT 
                'match_created' as type,
                'New AI match created' as message,
                match_at as timestamp
            FROM ai_matches 
            ORDER BY match_at DESC 
            LIMIT 2
        )
        UNION ALL
        (
            SELECT 
                'item_resolved' as type,
                CONCAT('Item resolved: ', title) as message,
                created_at as timestamp
            FROM items 
            WHERE status IN ('claimed', 'returned')
            ORDER BY created_at DESC 
            LIMIT 2
        )
        ORDER BY timestamp DESC 
        LIMIT 10
    """)
    recent_activity_raw = cursor.fetchall()
    
    cursor.close()
    
    # Convert datetime objects to ISO format strings for JSON serialization
    recent_activity = []
    for activity in recent_activity_raw:
        recent_activity.append({
            'type': activity['type'],
            'message': activity['message'],
            'timestamp': activity['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if activity['timestamp'] else None
        })
    
    return jsonify({
        'success': True,
        'stats': {
            'total_users': total_users,
            'total_items': total_items,
            'total_matches': total_matches,
            'total_revenue': total_revenue
        },
        'item_counts': item_counts,
        'monthly_data': monthly_data,
        'recent_activity': recent_activity,
        'period': period_int
    })


# Add this route after your existing /api/admin/analytics route in app.py

@app.route('/api/admin/reports')
def admin_reports():
    """Generate reports with date range filtering"""
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        report_type = request.args.get('type', 'all')
        
        # Validate dates
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Start and end dates are required'}), 400
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            # Add time to end date to include the entire day
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Base query for items
        if report_type == 'users':
    # User activity report
            users_query = """
                SELECT 
                    u.id,
                    u.first_name,
                    u.last_name,
                    u.email,
                    u.phone,
                    u.created_at as user_created_at,
                    u.profile_picture,
                    u.is_active,
                    COUNT(DISTINCT i.id) as total_items,
                    SUM(CASE WHEN i.type = 'lost' AND i.payment_status = 'paid' THEN 1 ELSE 0 END) as paid_lost_items,
                    SUM(CASE WHEN i.type = 'lost' AND (i.payment_status IS NULL OR i.payment_status != 'paid') THEN 1 ELSE 0 END) as unpaid_lost_items,
                    SUM(CASE WHEN i.type = 'found' THEN 1 ELSE 0 END) as found_items
                FROM users u
                LEFT JOIN items i ON u.id = i.user_id AND i.created_at BETWEEN %s AND %s
                WHERE u.created_at BETWEEN %s AND %s
                AND u.email != 'admin@reunited.com'  # EXCLUDE ADMIN
                GROUP BY u.id
                ORDER BY u.created_at DESC
            """
            cursor.execute(users_query, (start_dt, end_dt, start_dt, end_dt))
            items = cursor.fetchall()
            
            # Summary for users report
            summary_query = """
                SELECT 
                    COUNT(DISTINCT u.id) as total_users,
                    COUNT(DISTINCT i.id) as total_items,
                    SUM(CASE WHEN i.type = 'lost' AND i.payment_status = 'paid' THEN 20 ELSE 0 END) as total_revenue
                FROM users u
                LEFT JOIN items i ON u.id = i.user_id AND i.created_at BETWEEN %s AND %s
                WHERE u.created_at BETWEEN %s AND %s
                AND u.email != 'admin@reunited.com'  # EXCLUDE ADMIN FROM COUNT TOO
            """
            cursor.execute(summary_query, (start_dt, end_dt, start_dt, end_dt))
            summary = cursor.fetchone()
            
            # Summary for users report
            summary_query = """
                SELECT 
                    COUNT(DISTINCT u.id) as total_users,
                    COUNT(DISTINCT i.id) as total_items,
                    SUM(CASE WHEN i.type = 'lost' AND i.payment_status = 'paid' THEN 20 ELSE 0 END) as total_revenue
                FROM users u
                LEFT JOIN items i ON u.id = i.user_id AND i.created_at BETWEEN %s AND %s
                WHERE u.created_at BETWEEN %s AND %s
            """
            cursor.execute(summary_query, (start_dt, end_dt, start_dt, end_dt))
            summary = cursor.fetchone()
            
        elif report_type == 'revenue':
            # Revenue report (only paid lost items)
            revenue_query = """
                SELECT 
                    i.id,
                    i.title,
                    i.description,
                    i.category,
                    i.type,
                    i.date_reported,
                    i.location_reported,
                    i.status,
                    i.payment_status,
                    i.created_at,
                    u.first_name as user_first_name,
                    u.last_name as user_last_name,
                    u.email as user_email,
                    u.phone as user_phone,
                    u.created_at as user_created_at,
                    20 as amount
                FROM items i
                JOIN users u ON i.user_id = u.id
                WHERE i.created_at BETWEEN %s AND %s
                AND i.type = 'lost'
                AND i.payment_status = 'paid'
                ORDER BY i.created_at DESC
            """
            cursor.execute(revenue_query, (start_dt, end_dt))
            items = cursor.fetchall()
            
            # Summary for revenue report
            summary_query = """
                SELECT 
                    COUNT(*) as total_items,
                    COUNT(*) * 20 as total_revenue,
                    COUNT(DISTINCT user_id) as unique_payers
                FROM items
                WHERE created_at BETWEEN %s AND %s
                AND type = 'lost'
                AND payment_status = 'paid'
            """
            cursor.execute(summary_query, (start_dt, end_dt))
            summary = cursor.fetchone()
            
            # Summary for revenue report
            summary_query = """
                SELECT 
                    COUNT(*) as total_items,
                    COUNT(*) * 20 as total_revenue,
                    COUNT(DISTINCT user_id) as unique_payers
                FROM items
                WHERE created_at BETWEEN %s AND %s
                AND type = 'lost'
                AND payment_status = 'paid'
            """
            cursor.execute(summary_query, (start_dt, end_dt))
            summary = cursor.fetchone()
            
        else:
            # Items report (default)
            base_query = """
                SELECT 
                    i.*,
                    u.first_name as user_first_name,
                    u.last_name as user_last_name,
                    u.email as user_email,
                    u.phone as user_phone,
                    u.created_at as user_created_at,
                    img.file_path as image_path
                FROM items i
                JOIN users u ON i.user_id = u.id
                LEFT JOIN images img ON i.id = img.item_id
                WHERE i.created_at BETWEEN %s AND %s
            """
            
            params = [start_dt, end_dt]
            
            # Add type filter if specified
            if report_type == 'lost':
                base_query += " AND i.type = 'lost'"
            elif report_type == 'found':
                base_query += " AND i.type = 'found'"
            elif report_type == 'resolved':
                base_query += " AND i.status IN ('claimed', 'returned')"
            
            base_query += " ORDER BY i.created_at DESC"
            
            cursor.execute(base_query, params)
            items = cursor.fetchall()
            
            # Summary for items report
            summary_query = """
                SELECT 
                    COUNT(*) as total_items,
                    SUM(CASE WHEN type = 'lost' THEN 1 ELSE 0 END) as lost_items,
                    SUM(CASE WHEN type = 'found' THEN 1 ELSE 0 END) as found_items,
                    SUM(CASE WHEN status IN ('claimed', 'returned') THEN 1 ELSE 0 END) as resolved_items,
                    SUM(CASE WHEN payment_status = 'paid' AND type = 'lost' THEN 20 ELSE 0 END) as total_revenue
                FROM items
                WHERE created_at BETWEEN %s AND %s
            """
            cursor.execute(summary_query, (start_dt, end_dt))
            summary = cursor.fetchone()
        
        cursor.close()
        
        # Format dates and clean data for JSON response
        formatted_items = []
        for item in items:
            formatted_item = {}
            for key, value in item.items():
                if isinstance(value, datetime):
                    # Format for Excel compatibility
                    if key in ['created_at', 'date_reported', 'user_created_at', 'payment_date']:
                        # ISO format is best for Excel
                        formatted_item[key] = value.isoformat()
                    else:
                        formatted_item[key] = value.strftime('%Y-%m-%d')
                elif key == 'image_path' and value:
                    # Clean image path
                    formatted_item[key] = value.replace("\\", "/")
                    if formatted_item[key].startswith("static/"):
                        formatted_item[key] = formatted_item[key][7:]
                elif value is None:
                    formatted_item[key] = ''
                else:
                    formatted_item[key] = value
            formatted_items.append(formatted_item)
        
        # Format summary
        formatted_summary = {
            'start_date': start_date,
            'end_date': end_date,
            'report_type': report_type,
            'total_items': summary.get('total_items', 0) or 0,
            'lost_items': summary.get('lost_items', 0) or 0,
            'found_items': summary.get('found_items', 0) or 0,
            'resolved_items': summary.get('resolved_items', 0) or 0,
            'total_revenue': summary.get('total_revenue', 0) or 0,
            'total_users': summary.get('total_users', 0) or 0,
            'unique_payers': summary.get('unique_payers', 0) or 0
        }
        
        return jsonify({
            'success': True,
            'summary': formatted_summary,
            'items': formatted_items
        })
        
    except Exception as e:
        print(f"Error in admin reports API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

    
# API route for searching items - FIXED
@app.route('/api/admin/search-items')
def admin_search_items():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        search_query = request.args.get('q', '')
        item_type = request.args.get('type', '')
        status = request.args.get('status', '')
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        query = """
            SELECT 
                i.*,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.profile_picture,  # ✅ MAKE SURE THIS IS INCLUDED
                img.file_path as image_path
            FROM items i
            LEFT JOIN users u ON i.user_id = u.id
            LEFT JOIN images img ON i.id = img.item_id
            WHERE 1=1
        """
        params = []
        
        if search_query:
            query += " AND (i.title LIKE %s OR i.description LIKE %s OR u.first_name LIKE %s OR u.last_name LIKE %s)"
            params.extend([f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
        
        if item_type:
            query += " AND i.type = %s"
            params.append(item_type)
        
        if status:
            query += " AND i.status = %s"
            params.append(status)
        
        query += " ORDER BY i.created_at DESC LIMIT 100"
        
        cursor.execute(query, params)
        items = cursor.fetchall()
        cursor.close()
        
        # Fix image paths for display
        for item in items:
            if item['image_path']:
                # Just normalize slashes
                item['image_path'] = item['image_path'].replace("\\", "/")
                # ✅ REMOVED ALL THE COMPLEX PATH MANIPULATION
        
        return jsonify({
            'success': True,
            'items': items
        })
        
    except Exception as e:
        print(f"Error in admin search: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Fixed admin routes with correct template names
@app.route('/admin/users')
def admin_users():
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    return render_template('adminusers.html', active_page='admin_users')  # ✅ Correct filename
@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
def delete_admin_user(user_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Prevent deleting admin user (ID 20)
        if user_id == 20:
            return jsonify({'success': False, 'error': 'Cannot delete admin user'}), 400
        
        # Check if user exists
        cursor.execute("SELECT id, first_name, last_name, profile_picture FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Delete user's items and images first
        cursor.execute("SELECT id FROM items WHERE user_id = %s", (user_id,))
        items = cursor.fetchall()
        
        for item in items:
            # Delete item images
            cursor.execute("SELECT file_path FROM images WHERE item_id = %s", (item['id'],))
            images = cursor.fetchall()
            
            for image in images:
                if image['file_path'] and os.path.exists(image['file_path']):
                    try:
                        os.remove(image['file_path'])
                    except Exception as e:
                        print(f"Warning: Could not delete image file {image['file_path']}: {e}")
            
            cursor.execute("DELETE FROM images WHERE item_id = %s", (item['id'],))
        
        # Delete user's items
        cursor.execute("DELETE FROM items WHERE user_id = %s", (user_id,))
        
        # Delete user's profile picture
        if user['profile_picture']:
            profile_path = os.path.join(app.config['UPLOAD_FOLDER'], user['profile_picture'])
            if os.path.exists(profile_path):
                try:
                    os.remove(profile_path)
                except Exception as e:
                    print(f"Warning: Could not delete profile picture {profile_path}: {e}")
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': f'User {user["first_name"]} {user["last_name"]} deleted successfully'
        })
        
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/items')
def admin_items():
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    return render_template('adminitems.html', active_page='admin_items')  # ✅ Correct filename

@app.route('/api/admin/items/<int:item_id>', methods=['DELETE'])
def delete_admin_item(item_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Check if item exists and get user details
        cursor.execute("""
            SELECT i.id, i.title, i.user_id, i.type, 
                   u.email, u.first_name, u.last_name, u.profile_picture 
            FROM items i 
            JOIN users u ON i.user_id = u.id 
            WHERE i.id = %s""", (item_id,))
        item = cursor.fetchone()
        
        if not item:
            cursor.close()
            return jsonify({'success': False, 'error': 'Item not found'}), 404
        
        # Delete item images first
        cursor.execute("SELECT file_path FROM images WHERE item_id = %s", (item_id,))
        images = cursor.fetchall()
        
        for image in images:
            if image['file_path'] and os.path.exists(image['file_path']):
                try:
                    os.remove(image['file_path'])
                except Exception as e:
                    print(f"Warning: Could not delete image file {image['file_path']}: {e}")
        
        # Delete images from database
        cursor.execute("DELETE FROM images WHERE item_id = %s", (item_id,))
        
        # Delete any matches involving this item
        cursor.execute("DELETE FROM ai_matches WHERE lost_item_id = %s OR found_item_id = %s", (item_id, item_id))
        
        # Delete the item
        cursor.execute("DELETE FROM items WHERE id = %s", (item_id,))
        
        # Send notification to the user
        send_item_deleted_notification(
            user_id=item['user_id'],
            item_title=item['title'],
            reason="data privacy and platform guidelines",
            cursor=cursor,
            mysql=mysql
        )
        
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': f'Item "{item["title"]}" deleted successfully. User has been notified.',
            'user_notified': True,
            'user_name': f"{item['first_name']} {item['last_name']}",
            'user_id': item['user_id']  # ✅ ADD THIS LINE
        })
        
    except Exception as e:
        print(f"Error deleting item: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
# Add this function near the top with other helper functions in app.py
def send_item_deleted_notification(user_id, item_title, reason="data privacy", cursor=None, mysql=None):
    """Send notification to user when their item is deleted"""
    try:
        message = f"Your item '{item_title}' has been deleted by admin due to {reason}."
        notification_type = "item_deleted"
        
        if cursor is None:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get user's email and name for email notification
        cursor.execute("SELECT email, first_name FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if user:
            # Insert notification in database
            cursor.execute("""INSERT INTO notifications (user_id, type, message, sent_at, is_read) 
                            VALUES (%s, %s, %s, NOW(), 0)""", 
                          (user_id, notification_type, message))
            
            # Send email notification
            try:
                msg = Message(
                    subject='Item Deleted - Reunited',
                    recipients=[user['email']],
                    html=f'''
                    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                        <div style="text-align: center; margin-bottom: 30px;">
                            <h1 style="color: #1E3A8A; margin-bottom: 10px;">🤝 Reunited</h1>
                            <h2 style="color: #374151; font-weight: normal;">Item Deletion Notice</h2>
                        </div>
                        
                        <div style="background: #f8f9fa; padding: 30px; border-radius: 10px;">
                            <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">
                                Hi {user['first_name']},
                            </p>
                            <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">
                                Your item <strong>"{item_title}"</strong> has been deleted by the Reunited admin team.
                            </p>
                            <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">
                                <strong>Reason:</strong> {reason}
                            </p>
                            
                            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 20px 0;">
                                <p style="color: #856404; margin: 0;">
                                    <i class="fas fa-exclamation-circle" style="margin-right: 8px;"></i>
                                    This action was taken to maintain data privacy and security on our platform.
                                </p>
                            </div>
                            
                            <p style="font-size: 14px; color: #6b7280; margin-top: 20px;">
                                If you believe this was a mistake or have any questions, please contact our support team.
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
            except Exception as e:
                print(f"Error sending email notification: {e}")
                # Continue even if email fails
            
            return True
        return False
    except Exception as e:
        print(f"Error sending deletion notification: {e}")
        return False

@app.route('/admin/matches')
def admin_matches():
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    return render_template('adminmatches.html', active_page='admin_matches')  # ✅ Correct filename




@app.route('/admin/analytics')
def admin_analytics():
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    
    # Set default dates (last 7 days)
    today = datetime.now()
    last_week = today - timedelta(days=7)
    
    return render_template('adminanalytics.html', 
                         active_page='admin_analytics',
                         default_start_date=last_week.strftime('%Y-%m-%d'),
                         default_end_date=today.strftime('%Y-%m-%d'))

@app.route('/api/admin/analytics')
def api_admin_analytics():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        period = request.args.get('period', '4')  # Default to 4 weeks
        
        # Validate period parameter
        try:
            period_int = int(period)
            if period_int not in [4, 8, 12]:
                period_int = 4
        except ValueError:
            period_int = 4
        
        # Key Metrics
        cursor.execute("SELECT COUNT(*) as total_users FROM users WHERE is_active = TRUE")
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute("SELECT COUNT(*) as total_items FROM items")
        total_items = cursor.fetchone()['total_items']
        
        cursor.execute("SELECT COUNT(*) as total_matches FROM ai_matches WHERE status = 'resolved'")
        total_matches = cursor.fetchone()['total_matches']
        
        cursor.execute("SELECT COALESCE(SUM(CASE WHEN payment_status = 'paid' THEN 20 ELSE 0 END), 0) as total_revenue FROM items WHERE type = 'lost'")
        total_revenue = cursor.fetchone()['total_revenue']
        
        # Weekly Items Data
        cursor.execute(f"""
            SELECT 
                CONCAT('Week ', WEEK(created_at), ' - ', DATE_FORMAT(MIN(created_at), '%b %d')) as week_name,
                COUNT(*) as total_items,
                SUM(CASE WHEN type = 'lost' THEN 1 ELSE 0 END) as lost_items,
                SUM(CASE WHEN type = 'found' THEN 1 ELSE 0 END) as found_items
            FROM items 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL {period_int} WEEK)
            GROUP BY YEAR(created_at), WEEK(created_at)
            ORDER BY MIN(created_at) ASC
            LIMIT {period_int}
        """)
        weekly_data = cursor.fetchall()
        
        # Item Distribution
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN type = 'lost' AND status NOT IN ('claimed', 'returned') THEN 1 ELSE 0 END) as lost_items,
                SUM(CASE WHEN type = 'found' AND status NOT IN ('claimed', 'returned') THEN 1 ELSE 0 END) as found_items,
                SUM(CASE WHEN status IN ('claimed', 'returned') THEN 1 ELSE 0 END) as resolved_items
            FROM items
        """)
        item_distribution = cursor.fetchone()
        
        # Success Rate (matches resolved vs total matches)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_matches,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_matches
            FROM ai_matches
        """)
        match_data = cursor.fetchone()
        success_rate = (match_data['resolved_matches'] / match_data['total_matches'] * 100) if match_data['total_matches'] > 0 else 0
        
        # Recent Activity (last 10 activities)
        cursor.execute("""
            (
                SELECT 
                    'user_registered' as type,
                    CONCAT('New user registered: ', first_name, ' ', last_name) as message,
                    created_at as timestamp
                FROM users 
                ORDER BY created_at DESC 
                LIMIT 3
            )
            UNION ALL
            (
                SELECT 
                    'item_posted' as type,
                    CONCAT('New ', type, ' item: ', title) as message,
                    created_at as timestamp
                FROM items 
                ORDER BY created_at DESC 
                LIMIT 3
            )
            UNION ALL
            (
                SELECT 
                    'match_created' as type,
                    'New AI match created' as message,
                    match_at as timestamp
                FROM ai_matches 
                ORDER BY match_at DESC 
                LIMIT 2
            )
            UNION ALL
            (
                SELECT 
                    'item_resolved' as type,
                    CONCAT('Item resolved: ', title) as message,
                    created_at as timestamp
                FROM items 
                WHERE status IN ('claimed', 'returned')
                ORDER BY created_at DESC 
                LIMIT 2
            )
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent_activity = cursor.fetchall()
        
        cursor.close()
        
        # Prepare chart data
        weekly_items = {
            'labels': [item['week_name'] for item in weekly_data],
            'lost': [item['lost_items'] for item in weekly_data],
            'found': [item['found_items'] for item in weekly_data]
        }
        
        return jsonify({
            'success': True,
            'metrics': {
                'total_users': total_users,
                'total_items': total_items,
                'total_matches': total_matches,
                'total_revenue': total_revenue
            },
            'charts': {
                'weekly_items': weekly_items,  # Changed from monthly_items to weekly_items
                'item_distribution': item_distribution,
                'success_rate': round(success_rate, 1)
            },
            'recent_activity': recent_activity
        })
        
    except Exception as e:
        print(f"Error in admin analytics API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ✅ CHANGED: Reports to Feedbacks
@app.route('/admin/feedbacks')
def admin_feedbacks():
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login_page'))
    return render_template('adminfeedbacks.html', active_page='admin_feedbacks')  # ✅ New filename

# ✅ REMOVED: admin_settings route (replaced with logout in sidebar)

# Add placeholder APIs for new admin pages
@app.route('/api/admin/users')
def api_admin_users():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get query parameters
        search_query = request.args.get('q', '')
        status_filter = request.args.get('status', '')
        
        # Build base query
        base_query = """
            SELECT 
                u.id, u.first_name, u.last_name, u.email, u.phone, 
                u.profile_picture, u.is_active, u.created_at,
                COUNT(i.id) as items_count
            FROM users u
            LEFT JOIN items i ON u.id = i.user_id
        """
        
        # Build WHERE conditions
        where_conditions = []
        params = []
        
        if search_query:
            where_conditions.append("(u.first_name LIKE %s OR u.last_name LIKE %s OR u.email LIKE %s OR u.phone LIKE %s)")
            params.extend([f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
        
        if status_filter == 'active':
            where_conditions.append("u.is_active = TRUE")
        elif status_filter == 'inactive':
            where_conditions.append("u.is_active = FALSE")
        
        # Construct final query
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        group_by = " GROUP BY u.id"
        order_by = " ORDER BY u.created_at DESC"
        
        final_query = base_query + where_clause + group_by + order_by
        
        print(f"Executing query: {final_query}")  # Debug
        print(f"With params: {params}")  # Debug
        
        cursor.execute(final_query, params)
        users = cursor.fetchall()
        cursor.close()
        
        print(f"Found {len(users)} users")  # Debug
        
        return jsonify({
            'success': True,
            'users': users
        })
        
    except Exception as e:
        print(f"Error in admin users API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/matches')
def api_admin_matches():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Get query parameters
        search_query = request.args.get('q', '')
        status_filter = request.args.get('status', '')
        confidence_filter = request.args.get('confidence', '')
        
        # Build query
        query = """
            SELECT 
                m.id, m.match_score, m.match_at, m.status,
                lost.id as lost_item_id, lost.title as lost_title,
                found.id as found_item_id, found.title as found_title,
                CONCAT(u1.first_name, ' ', u1.last_name) as lost_user_name,
                CONCAT(u2.first_name, ' ', u2.last_name) as found_user_name,
                lost_img.file_path as lost_image_path,
                found_img.file_path as found_image_path
            FROM ai_matches m
            JOIN items lost ON m.lost_item_id = lost.id
            JOIN items found ON m.found_item_id = found.id
            JOIN users u1 ON lost.user_id = u1.id
            JOIN users u2 ON found.user_id = u2.id
            LEFT JOIN images lost_img ON lost.id = lost_img.item_id
            LEFT JOIN images found_img ON found.id = found_img.item_id
            WHERE 1=1
        """
        params = []
        
        # Add search filters
        if search_query:
            query += " AND (lost.title LIKE %s OR found.title LIKE %s OR u1.first_name LIKE %s OR u1.last_name LIKE %s OR u2.first_name LIKE %s OR u2.last_name LIKE %s)"
            params.extend([f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
        
        # Add status filter
        if status_filter:
            query += " AND m.status = %s"
            params.append(status_filter)
        
        # Add confidence filter
        if confidence_filter == 'high':
            query += " AND m.match_score >= 0.8"
        elif confidence_filter == 'medium':
            query += " AND m.match_score >= 0.4 AND m.match_score < 0.8"
        elif confidence_filter == 'low':
            query += " AND m.match_score < 0.4"
        
        query += " ORDER BY m.match_at DESC"
        
        cursor.execute(query, params)
        matches = cursor.fetchall()
        cursor.close()
        
        # Process image paths exactly like in admin/items
        for match in matches:
            # Process lost item image path
            if match.get('lost_image_path'):
                match['lost_image_path'] = match['lost_image_path'].replace("\\", "/")
                # ✅ REMOVED ALL THE COMPLEX PATH MANIPULATION
            
            # Process found item image path
            if match.get('found_image_path'):
                match['found_image_path'] = match['found_image_path'].replace("\\", "/")
                # ✅ REMOVED ALL THE COMPLEX PATH MANIPULATION
        
        return jsonify({
            'success': True,
            'matches': matches
        })
        
    except Exception as e:
        print(f"Error in admin matches API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/matches/<int:match_id>/approve', methods=['PUT'])
def approve_admin_match(match_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Update match status to resolved
        cursor.execute("UPDATE ai_matches SET status = 'resolved' WHERE id = %s", (match_id,))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Match approved successfully'
        })
        
    except Exception as e:
        print(f"Error approving match: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/matches/<int:match_id>/reject', methods=['PUT'])
def reject_admin_match(match_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Update match status to rejected
        cursor.execute("UPDATE ai_matches SET status = 'rejected' WHERE id = %s", (match_id,))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Match rejected successfully'
        })
        
    except Exception as e:
        print(f"Error rejecting match: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/feedbacks')
def api_admin_feedbacks():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        cursor.execute("""
            SELECT 
                f.id, f.user_id, f.message, f.rating, f.is_public, f.created_at,
                u.first_name, u.last_name, u.email, u.profile_picture
            FROM feedback f
            JOIN users u ON f.user_id = u.id
            ORDER BY f.created_at DESC
        """)
        
        feedbacks = cursor.fetchall()
        cursor.close()
        
        return jsonify({
            'success': True,
            'feedbacks': feedbacks
        })
        
    except Exception as e:
        print(f"Error in admin feedbacks API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/feedbacks/<int:feedback_id>/visibility', methods=['PUT'])
def update_feedback_visibility(feedback_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        is_public = data.get('is_public', False)
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("UPDATE feedback SET is_public = %s WHERE id = %s", (is_public, feedback_id))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Feedback visibility updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating feedback visibility: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/feedbacks/<int:feedback_id>', methods=['DELETE'])
def delete_admin_feedback(feedback_id):
    if not session.get('is_admin'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Delete the feedback
        cursor.execute("DELETE FROM feedback WHERE id = %s", (feedback_id,))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Feedback deleted successfully'
        })
        
    except Exception as e:
        print(f"Error deleting feedback: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        i.status, i.created_at, i.date_reported, i.payment_status,
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

# Add this function to send password reset emails
def send_password_reset_email(email, otp, first_name):
    """Send password reset OTP via email"""
    try:
        msg = Message(
            subject='Password Reset - Reunited',
            recipients=[email],
            html=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #1E3A8A; margin-bottom: 10px;">🤝 Reunited</h1>
                    <h2 style="color: #374151; font-weight: normal;">Password Reset</h2>
                </div>
                
                <div style="background: #f8f9fa; padding: 30px; border-radius: 10px; text-align: center;">
                    <p style="font-size: 16px; color: #374151; margin-bottom: 20px;">
                        Hi {first_name},
                    </p>
                    <p style="font-size: 16px; color: #374151; margin-bottom: 30px;">
                        You requested to reset your password. Use the verification code below:
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
        print(f"Error sending password reset email: {e}")
        return False

# Add these routes to your Flask app
@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    """Send OTP for password reset"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'success': False, 'errors': ['Email is required']}), 400
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT id, first_name, email FROM users WHERE email = %s AND is_active = TRUE', (email,))
        user = cursor.fetchone()
        cursor.close()
        
        if not user:
            # Don't reveal if email exists or not for security
            return jsonify({
                'success': True, 
                'message': 'If your email is registered, you will receive a verification code shortly.'
            }), 200
        
        # Generate and store OTP
        otp = generate_otp()
        session['reset_otp'] = otp
        session['reset_email'] = email
        session['reset_otp_timestamp'] = datetime.now().timestamp()
        
        # Send OTP email
        if send_password_reset_email(email, otp, user['first_name']):
            return jsonify({
                'success': True, 
                'message': 'Verification code sent to your email!'
            }), 200
        else:
            return jsonify({'success': False, 'errors': ['Failed to send verification email. Please try again.']}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/verify-reset-otp', methods=['POST'])
def verify_reset_otp():
    """Verify OTP for password reset"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        entered_otp = data.get('otp_code', '').strip()
        
        if not email or not entered_otp:
            return jsonify({'success': False, 'errors': ['Email and OTP code are required']}), 400
        
        # Check if OTP exists in session
        if ('reset_otp' not in session or 'reset_email' not in session or 
            session.get('reset_email') != email):
            return jsonify({'success': False, 'errors': ['Invalid reset session. Please request a new code.']}), 400
        
        # Check OTP expiry (10 minutes)
        otp_timestamp = session.get('reset_otp_timestamp', 0)
        current_timestamp = datetime.now().timestamp()
        if current_timestamp - otp_timestamp > 600:  # 10 minutes
            session.pop('reset_otp', None)
            session.pop('reset_email', None)
            session.pop('reset_otp_timestamp', None)
            return jsonify({'success': False, 'errors': ['Verification code expired. Please request a new one.']}), 400
        
        # Verify OTP
        stored_otp = session.get('reset_otp')
        if entered_otp != stored_otp:
            return jsonify({'success': False, 'errors': ['Invalid verification code. Please try again.']}), 400
        
        # OTP is valid, mark as verified for password reset
        session['reset_verified'] = True
        session['reset_email'] = email
        
        return jsonify({
            'success': True, 
            'message': 'Verification successful! You can now reset your password.'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    """Reset password after OTP verification"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        new_password = data.get('new_password', '')
        
        if not email or not new_password:
            return jsonify({'success': False, 'errors': ['Email and new password are required']}), 400
        
        # Check if reset session is verified
        if (not session.get('reset_verified') or 
            session.get('reset_email') != email):
            return jsonify({'success': False, 'errors': ['Reset session not verified. Please start over.']}), 400
        
        # Password validation
        if len(new_password) < 8:
            return jsonify({'success': False, 'errors': ['Password must be at least 8 characters long']}), 400
        
        if not re.search(r'[A-Z]', new_password):
            return jsonify({'success': False, 'errors': ['Password must contain at least one uppercase letter']}), 400
        
        if not re.search(r'[a-z]', new_password):
            return jsonify({'success': False, 'errors': ['Password must contain at least one lowercase letter']}), 400
        
        if not re.search(r'\d', new_password):
            return jsonify({'success': False, 'errors': ['Password must contain at least one number']}), 400
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        cursor.execute('UPDATE users SET password = %s WHERE email = %s', (hashed_password, email))
        mysql.connection.commit()
        cursor.close()
        
        # Clear reset session
        session.pop('reset_otp', None)
        session.pop('reset_email', None)
        session.pop('reset_otp_timestamp', None)
        session.pop('reset_verified', None)
        
        return jsonify({
            'success': True, 
            'message': 'Password reset successfully! You can now log in with your new password.'
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500



# Update your login function to handle admin redirect
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
        
        # Check for admin credentials
        if username.lower() == 'admin@reunited.com' and password == 'Admin@123':
            # Create admin session
            session['user_id'] = 'admin'
            session['first_name'] = 'Admin'
            session['last_name'] = 'User'
            session['email'] = 'admin@reunited.com'
            session['full_name'] = 'Admin User'
            session['is_admin'] = True
            
            return jsonify({
                'success': True, 
                'message': 'Admin login successful!',
                'redirect': '/admindashboard',
                'is_admin': True
            }), 200
        
        # Check if user exists (by email or username) - regular user login
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s AND is_active = TRUE', (username.lower(),))
        user = cursor.fetchone()
        cursor.close()
        
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'success': False, 'errors': ['Invalid credentials']}), 401
        
        # Create session for regular user
        session['user_id'] = user['id']
        session['first_name'] = user['first_name']
        session['last_name'] = user['last_name']
        session['email'] = user['email']
        session['full_name'] = f"{user['first_name']} {user['last_name']}"
        session['phone'] = user['phone']
        session['profile_picture'] = user['profile_picture']
        session['is_admin'] = False
        
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
        
        # Check if user wants to remove profile picture
        remove_profile_picture = request.form.get('remove_profile_picture') == 'true'

        first_name, last_name = "", ""
        if " " in full_name:
            first_name, last_name = full_name.split(" ", 1)
        else:
            first_name = full_name
            last_name = ""

        # Handle profile picture
        profile_picture_filename = session.get('profile_picture')
        
        if remove_profile_picture:
            # User wants to remove existing picture
            if profile_picture_filename:
                old_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_picture_filename)
                if os.path.exists(old_path):
                    os.remove(old_path)
                profile_picture_filename = None
                flash('Profile picture removed!', 'success')
        elif 'profile_picture' in request.files:
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
                flash('Profile picture updated!', 'success')

        # Update SQL
        if password:
            hashed_password = generate_password_hash(password)
            cursor.execute('''UPDATE users SET first_name=%s, last_name=%s, email=%s, phone=%s, password=%s, profile_picture=%s WHERE id=%s''', 
                         (first_name, last_name, email, phone, hashed_password, profile_picture_filename, session['user_id']))
        else:
            cursor.execute('''UPDATE users SET first_name=%s, last_name=%s, email=%s, phone=%s, profile_picture=%s WHERE id=%s''', 
                         (first_name, last_name, email, phone, profile_picture_filename, session['user_id']))

        mysql.connection.commit()

        # Update session
        session['first_name'] = first_name
        session['last_name'] = last_name
        session['full_name'] = f"{first_name} {last_name}".strip()
        session['email'] = email
        session['phone'] = phone
        session['profile_picture'] = profile_picture_filename

        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))

    # GET request - refresh user data from DB
    cursor.execute('SELECT * FROM users WHERE id=%s', (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()

    # Ensure session has all data
    session['phone'] = user['phone']
    session['profile_picture'] = user['profile_picture']

    return render_template('profile.html', active_page='profile')


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

        # insert item with payment_status = 'pending'
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, reward, status, payment_status) VALUES (%s, %s, %s, %s, 'lost', %s, %s, %s, %s, 'pending')''', (session['user_id'], title, description, category, date_reported, location, reward, "Not Yet Found"))

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

                    # Extract features
                    try:
                        features = extract_features(filepath)
                        print(f"✅ Image features extracted successfully")
                    except Exception as e:
                        print(f"❌ Error extracting features: {e}")
                        features = None

                    # insert into images table with features
                    cursor.execute('''INSERT INTO images (item_id, file_path, ai_features) VALUES (%s, %s, %s)''', 
                                  (item_id, filepath, str(features.tolist()) if features is not None else None))
                    mysql.connection.commit()
        
        # ❌ REMOVED: Do NOT auto-match here for lost items
        # Auto-matching will happen in payment-success route after payment
        
        flash('Lost item submitted successfully! Please complete payment to make it visible.', 'success')
        cursor.close()
        return redirect(url_for('posted'))

    # GET → show items with reporter info (only paid items that are NOT claimed)
    cursor.execute("""SELECT i.*, img.file_path AS image_path, u.first_name, u.last_name, u.profile_picture 
                     FROM items i 
                     LEFT JOIN images img ON i.id = img.item_id 
                     LEFT JOIN users u ON i.user_id = u.id 
                     WHERE i.type='lost' 
                     AND i.payment_status = 'paid' 
                     AND i.status NOT IN ('claimed', 'Claimed', 'returned', 'Returned', 'resolved', 'Resolved')
                     ORDER BY i.created_at DESC""")
    
    lost_items = cursor.fetchall()
    
    # Get the item_id from URL parameter if present
    show_item_id = request.args.get('show_item')
    
    # Fix image paths for display
    for item in lost_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            # ✅ REMOVED THE LINE
    
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

        # insert item with NULL reward (found items don't offer rewards)
        cursor.execute('''INSERT INTO items (user_id, title, description, category, type, date_reported, location_reported, reward, status) VALUES (%s, %s, %s, %s, 'found', %s, %s, NULL, %s)''', 
                      (session['user_id'], title, description, category, date_reported, location, "Unclaimed"))

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
                    cursor.execute('''INSERT INTO images (item_id, file_path, ai_features) VALUES (%s, %s, %s)''', 
                                  (item_id, filepath, str(features.tolist())))
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

    # GET → show items with reporter info (only unclaimed/not returned items)
    cursor.execute("""SELECT i.*, img.file_path AS image_path, u.first_name, u.last_name, u.profile_picture 
                     FROM items i 
                     LEFT JOIN images img ON i.id = img.item_id 
                     LEFT JOIN users u ON i.user_id = u.id 
                     WHERE i.type='found' 
                     AND i.status NOT IN ('returned', 'Returned', 'claimed', 'Claimed', 'resolved', 'Resolved')
                     ORDER BY i.created_at DESC""")
    
    found_items = cursor.fetchall()
    
    # Get the item_id from URL parameter if present
    show_item_id = request.args.get('show_item')
    
    # Fix image paths for display
    for item in found_items:
        if item['image_path']:
            item['image_path'] = item['image_path'].replace("\\", "/")
            # ✅ REMOVED THE LINE
    
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
            # ✅ REMOVED THE LINE
    
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
                'payment_status': item.get('payment_status', 'pending'),
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
        
        # 1. Verify ownership AND get current item type
        cursor.execute("SELECT user_id, type FROM items WHERE id = %s", (item_id,))
        item = cursor.fetchone()
        
        if not item or item['user_id'] != session['user_id']:
            return jsonify({'success': False, 'errors': ['Not authorized']}), 403
        
        # 2. Update item
        cursor.execute("""UPDATE items SET 
            title = %s, description = %s, category = %s, 
            location_reported = %s, reward = %s 
            WHERE id = %s""", 
            (data.get('title'), data.get('description'), data.get('category'),
             data.get('location_reported'), data.get('reward'), item_id))
        
        mysql.connection.commit()
        
        # 3. CHECK: Only trigger auto-match if item is ACTIVE (not claimed/returned)
        cursor.execute("SELECT status FROM items WHERE id = %s", (item_id,))
        current_status = cursor.fetchone()['status']
        
        active_statuses = ['Not Yet Found', 'Unclaimed', 'active', 'pending']
        if current_status in active_statuses:
            print(f"🔄 Item {item_id} was edited and is active - triggering re-matching...")
            
            # 4. Get updated details for matching
            updated_details = {
                "title": data.get('title'),
                "description": data.get('description'),
                "category": data.get('category'),
                "location": data.get('location_reported')
            }
            
            # 5. Get image features if exists
            cursor.execute("SELECT ai_features FROM images WHERE item_id = %s", (item_id,))
            image_data = cursor.fetchone()
            
            features = None
            if image_data and image_data['ai_features']:
                features = np.array(eval(image_data['ai_features']))
            
            # 6. Trigger auto-match with updated details
            auto_match(
                item_id, 
                item['type'],  # 'lost' or 'found'
                features, 
                updated_details, 
                cursor, 
                mysql, 
                mail
            )
            
            print(f"✅ Re-matching completed for edited item {item_id}")
        else:
            print(f"⚠️ Item {item_id} is {current_status} - skipping re-matching")
        
        cursor.close()
        
        return jsonify({
            'success': True, 
            'message': 'Item updated successfully' + (' (and re-matched)' if current_status in active_statuses else '')
        }), 200
        
    except Exception as e:
        print(f"❌ Error updating item: {str(e)}")
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500






# ========== FIXED PAYMONGO PAYMENT ROUTES ==========

@app.route('/api/initiate-payment/<int:item_id>', methods=['POST'])
def initiate_payment(item_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Verify item belongs to user and payment is pending
        cursor.execute("SELECT id, title, user_id, payment_status FROM items WHERE id = %s AND user_id = %s", 
                      (item_id, session['user_id']))
        item = cursor.fetchone()
        cursor.close()
        
        if not item:
            return jsonify({'success': False, 'errors': ['Item not found or not authorized']}), 404
        
        if item['payment_status'] == 'paid':
            return jsonify({'success': False, 'errors': ['Payment already completed']}), 400
        
        # Create PayMongo Checkout Session
        auth_string = base64.b64encode(f"{PAYMONGO_SECRET_KEY}:".encode()).decode()
        headers = {
            'Authorization': f'Basic {auth_string}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Construct success and cancel URLs
        base_url = request.host_url.rstrip('/')
        success_url = f"{base_url}/payment-success?item_id={item_id}"
        cancel_url = f"{base_url}/posted"
        
        print(f"🔐 Creating checkout session for item {item_id}")
        print(f"📍 Base URL: {base_url}")
        print(f"✅ Success URL: {success_url}")
        print(f"❌ Cancel URL: {cancel_url}")
        
        checkout_data = {
            "data": {
                "attributes": {
                    "line_items": [
                        {
                            "currency": "PHP",
                            "amount": 2000,  # 20 PHP in centavos
                            "name": f"Lost Item Posting: {item['title'][:50]}",
                            "quantity": 1
                        }
                    ],
                    "payment_method_types": ["card", "gcash", "paymaya"],
                    "success_url": success_url,
                    "cancel_url": cancel_url,
                    "description": f"Payment for posting lost item: {item['title'][:100]}",
                    "metadata": {
                        "item_id": str(item_id),
                        "user_id": str(session['user_id'])
                    }
                }
            }
        }
        
        print(f"📦 Checkout data: {json.dumps(checkout_data, indent=2)}")
        
        # Create checkout session with PayMongo
        response = requests.post(
            'https://api.paymongo.com/v1/checkout_sessions',
            headers=headers,
            json=checkout_data,
            timeout=30
        )
        
        print(f"📡 PayMongo Response Status: {response.status_code}")
        print(f"📡 PayMongo Response Headers: {response.headers}")
        print(f"📡 PayMongo Response Body: {response.text}")
        
        if response.status_code in [200, 201]:
            checkout_session = response.json()
            
            # Extract the correct fields from PayMongo response
            session_data = checkout_session.get('data', {})
            attributes = session_data.get('attributes', {})
            
            checkout_url = attributes.get('checkout_url')
            session_id = session_data.get('id')
            
            print(f"✅ Checkout URL: {checkout_url}")
            print(f"✅ Session ID: {session_id}")
            
            if not checkout_url or not session_id:
                print(f"❌ Missing data in response!")
                print(f"Full response: {json.dumps(checkout_session, indent=2)}")
                return jsonify({
                    'success': False, 
                    'errors': ['Invalid response from payment service']
                }), 500
            
            return jsonify({
                'success': True,
                'checkout_url': checkout_url,
                'session_id': session_id
            }), 200
        else:
            # Handle error response
            error_data = response.json() if response.text else {}
            error_messages = []
            
            if 'errors' in error_data:
                for err in error_data['errors']:
                    detail = err.get('detail', 'Unknown error')
                    error_messages.append(detail)
            else:
                error_messages.append(f"Payment service error: {response.status_code}")
            
            error_msg = ', '.join(error_messages)
            print(f"❌ PayMongo API error: {error_msg}")
            print(f"❌ Full error response: {response.text}")
            
            return jsonify({
                'success': False, 
                'errors': [error_msg]
            }), 500
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
        return jsonify({
            'success': False, 
            'errors': ['Request timeout. Please try again.']
        }), 500
    except requests.exceptions.RequestException as e:
        print(f"❌ Request exception: {str(e)}")
        return jsonify({
            'success': False, 
            'errors': [f'Network error: {str(e)}']
        }), 500
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ Exception: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'errors': [error_msg]
        }), 500


@app.route('/payment-success')
def payment_success():
    """Handle successful payment callback from PayMongo"""
    if 'user_id' not in session:
        flash('Session expired. Please log in again.', 'danger')
        return redirect(url_for('login_page'))
    
    item_id = request.args.get('item_id')
    
    if not item_id:
        flash('Invalid payment confirmation.', 'danger')
        return redirect(url_for('posted'))
    
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # Verify item belongs to user
        cursor.execute("SELECT id, title, payment_status FROM items WHERE id = %s AND user_id = %s", 
                      (item_id, session['user_id']))
        item = cursor.fetchone()
        
        if not item:
            flash('Item not found.', 'danger')
            cursor.close()
            return redirect(url_for('posted'))
        
        # Update payment status to paid
        cursor.execute("UPDATE items SET payment_status = 'paid' WHERE id = %s", (item_id,))
        mysql.connection.commit()
        
        # Update dashboard stats
        update_dashboard_stats(session['user_id'], cursor, mysql)
        
        # ✅ Trigger auto-matching for ALL paid items (with or without images)
        print(f"🔄 Triggering auto-matching for paid item {item_id}")
        
        # Get item details for matching
        cursor.execute("""
            SELECT i.*, img.ai_features 
            FROM items i 
            LEFT JOIN images img ON i.id = img.item_id 
            WHERE i.id = %s
        """, (item_id,))
        
        paid_item = cursor.fetchone()
        
        if paid_item:
            print(f"📋 Paid item details: {paid_item['title']}, Category: {paid_item['category']}")
            
            # Prepare item details for matching
            details = {
                "title": paid_item['title'],
                "description": paid_item['description'],
                "category": paid_item['category'],
                "location": paid_item['location_reported']
            }
            
            # Handle features (could be None if no image)
            features = None
            if paid_item['ai_features']:
                try:
                    features = np.array(eval(paid_item['ai_features']))
                    print(f"🖼️ Item has image features")
                except:
                    features = None
                    print(f"⚠️ Could not parse image features")
            else:
                print(f"📝 Item has NO image (text-only matching)")
            
            print(f"🔍 Starting auto-match for: {details['title']}")
            # Trigger auto-matching (pass None for features if no image)
            auto_match(item_id, 'lost', features, details, cursor, mysql, mail)
            print(f"✅ Auto-matching completed for item {item_id}")
        else:
            print(f"❌ Could not retrieve paid item details for {item_id}")
        
        cursor.close()
        
        print(f"✅ Payment successful for item {item_id}")
        flash(f'✅ Payment successful! Your lost item "{item["title"]}" is now visible to everyone.', 'success')
        return redirect(url_for('posted'))
        
    except Exception as e:
        print(f"❌ Error in payment success: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error confirming payment. Please contact support.', 'danger')
        return redirect(url_for('posted'))


@app.route('/api/verify-payment/<int:item_id>', methods=['POST'])
def verify_payment(item_id):
    """Optional: Verify payment status with PayMongo"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'errors': ['Not authenticated']}), 401
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'errors': ['Session ID required']}), 400
        
        # Retrieve checkout session from PayMongo
        auth_string = base64.b64encode(f"{PAYMONGO_SECRET_KEY}:".encode()).decode()
        headers = {
            'Authorization': f'Basic {auth_string}',
            'Accept': 'application/json'
        }
        
        response = requests.get(
            f'https://api.paymongo.com/v1/checkout_sessions/{session_id}',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            session_data = response.json()
            payment_intent = session_data.get('data', {}).get('attributes', {}).get('payment_intent')
            
            if payment_intent:
                payment_status = payment_intent.get('attributes', {}).get('status')
                
                cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                
                if payment_status == 'succeeded':
                    cursor.execute("UPDATE items SET payment_status = 'paid' WHERE id = %s AND user_id = %s", 
                                  (item_id, session['user_id']))
                    mysql.connection.commit()
                    cursor.close()
                    
                    return jsonify({
                        'success': True,
                        'message': 'Payment verified successfully'
                    }), 200
                else:
                    cursor.execute("UPDATE items SET payment_status = 'failed' WHERE id = %s AND user_id = %s", 
                                  (item_id, session['user_id']))
                    mysql.connection.commit()
                    cursor.close()
                    
                    return jsonify({
                        'success': False,
                        'errors': [f'Payment not completed: {payment_status}']
                    }), 400
            else:
                return jsonify({
                    'success': False,
                    'errors': ['Payment intent not found']
                }), 400
        else:
            return jsonify({
                'success': False,
                'errors': ['Failed to verify payment with PayMongo']
            }), 500
            
    except Exception as e:
        print(f"❌ Verification error: {str(e)}")
        return jsonify({'success': False, 'errors': [str(e)]}), 500
    

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
            m['lost_image'] = m['lost_image'].replace("\\", "/")
            # ✅ REMOVED THE LINE
        
        if m['found_image']:
            m['found_image'] = m['found_image'].replace("\\", "/")
            # ✅ REMOVED THE LINE

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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)