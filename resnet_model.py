import numpy as np
import re
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from utils import send_email

# -----------------------------
# Load ResNet50 once at startup
# -----------------------------
base_model = ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

# -----------------------------
# Image Feature Extraction
# -----------------------------
def extract_features(img_path):
    """Extract deep vector features from an image using ResNet50"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

def cosine_similarity(a, b):
    """Cosine similarity between two vectors"""
    if a is None or b is None:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Text Feature Matching (TF-IDF)
# -----------------------------
def clean_text(text):
    """Clean text by removing special characters and converting to lowercase"""
    if not text:
        return ""
    return re.sub(r'[^a-zA-Z0-9 ]', '', str(text).lower()).strip()

def text_similarity(text1, text2):
    """Calculate TF-IDF based cosine similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    docs = [clean_text(text1), clean_text(text2)]
    
    if not docs[0] or not docs[1]:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer().fit_transform(docs)
        vectors = vectorizer.toarray()
        return cos_sim([vectors[0]], [vectors[1]])[0][0]
    except:
        return 0.0

def enhanced_text_similarity(details1, details2):
    """
    Enhanced text matching with weighted components:
    - Category exact match: 20% bonus
    - Location similarity: 20% weight
    - Title + Description: 60% weight
    """
    
    category_bonus = 0.0
    if details1.get('category') and details2.get('category'):
        if clean_text(details1['category']) == clean_text(details2['category']):
            category_bonus = 0.2
    
    location_score = 0.0
    loc1 = details1.get('location', '')
    loc2 = details2.get('location', '')
    if loc1 and loc2:
        location_score = text_similarity(loc1, loc2) * 0.2
    
    content1 = f"{details1.get('title', '')} {details1.get('description', '')}"
    content2 = f"{details2.get('title', '')} {details2.get('description', '')}"
    content_score = text_similarity(content1, content2) * 0.6
    
    final_text_score = min(1.0, category_bonus + location_score + content_score)
    
    return final_text_score

def get_match_confidence_level(score):
    """Convert numeric score to confidence level description"""
    if score >= 0.8:
        return "Very High", "🟢"
    elif score >= 0.65:
        return "High", "🟡"
    elif score >= 0.5:
        return "Moderate", "🟠"
    else:
        return "Low", "🔴"

# -----------------------------
# Combined Auto Match with Dynamic Weights
# -----------------------------
def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql, mail, threshold=None):
    """
    Compare new item against opposite type (lost vs found).
    Uses dynamic weights based on whether items have images.
    """
    opposite_type = 'found' if type_ == 'lost' else 'lost'

    if opposite_type == 'lost':
        cursor.execute('''
            SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
                   img.ai_features,
                   CONCAT(u.first_name, ' ', u.last_name) AS fullname,
                   u.email, u.phone
            FROM items i
            LEFT JOIN images img ON i.id = img.item_id
            JOIN users u ON i.user_id = u.id
            WHERE i.type = 'lost'
            AND i.status NOT IN ('claimed', 'returned', 'resolved', 'Claimed', 'Returned', 'Resolved')
            AND i.payment_status = 'paid'
        ''')
    else:
        cursor.execute('''
            SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
                   img.ai_features,
                   CONCAT(u.first_name, ' ', u.last_name) AS fullname,
                   u.email, u.phone
            FROM items i
            LEFT JOIN images img ON i.id = img.item_id
            JOIN users u ON i.user_id = u.id
            WHERE i.type = 'found'
            AND i.status NOT IN ('claimed', 'returned', 'resolved', 'Claimed', 'Returned', 'Resolved')
        ''')

    others = cursor.fetchall()

    print(f"🔍 Comparing new {type_} item {new_item_id} against {len(others)} active {opposite_type} items")
    
    new_item_has_image = new_features is not None
    
    if threshold is None:
        threshold = 0.4 if new_item_has_image else 0.5
    
    for other in others:
        try:
            other_has_image = other['ai_features'] is not None
            
            if new_item_has_image and other_has_image:
                image_weight = 0.7
                text_weight = 0.3
            elif new_item_has_image and not other_has_image:
                image_weight = 0.6
                text_weight = 0.4
            else:
                image_weight = 0.0
                text_weight = 1.0
            
            print(f"Weights for item {other['id']}: Image={image_weight}, Text={text_weight}")
            
            image_score = 0.0
            if new_item_has_image and other_has_image:
                try:
                    other_features = np.array(eval(other['ai_features']))
                    image_score = cosine_similarity(new_features, other_features)
                except:
                    image_score = 0.0
            
            other_details = {
                'title': other['title'],
                'description': other['description'],
                'category': other['category'],
                'location': other['location_reported']
            }
            text_score = enhanced_text_similarity(new_details, other_details)
            
            final_score = (image_weight * image_score) + (text_weight * text_score)

            print(f"Comparing: {new_details['title']} (ID:{new_item_id}) vs {other['title']} (ID:{other['id']})")
            print(f"  Image: {image_score:.2f} (weight={image_weight:.1f}), Text: {text_score:.2f} (weight={text_weight:.1f}), Final: {final_score:.2f}, Threshold: {threshold}")

            if final_score >= threshold:
                lost_id = new_item_id if type_ == 'lost' else other['id']
                found_id = other['id'] if type_ == 'lost' else new_item_id

                cursor.execute('''
                    SELECT id FROM ai_matches 
                    WHERE (lost_item_id = %s AND found_item_id = %s)
                    OR (lost_item_id = %s AND found_item_id = %s)
                    OR (lost_item_id = %s AND found_item_id = %s)
                ''', (lost_id, found_id, found_id, lost_id, new_item_id, other['id']))
                
                existing_match = cursor.fetchone()
                
                if existing_match:
                    print(f"⚠️ Match already exists between items {lost_id} and {found_id} (Match ID: {existing_match['id']}), skipping...")
                    continue

                cursor.execute('''
                    INSERT INTO ai_matches (lost_item_id, found_item_id, match_score, status)
                    VALUES (%s, %s, %s, 'pending')
                ''', (lost_id, found_id, round(float(final_score), 2)))
                mysql.connection.commit()

                match_id = cursor.lastrowid 

                cursor.execute("""
                    SELECT id, CONCAT(first_name, ' ', u.last_name) AS fullname, u.email, u.phone
                    FROM users u WHERE u.id = (SELECT user_id FROM items WHERE id = %s)
                """, (lost_id,))
                lost_user = cursor.fetchone()

                cursor.execute("""
                    SELECT id, CONCAT(first_name, ' ', u.last_name) AS fullname, u.email, u.phone
                    FROM users u WHERE u.id = (SELECT user_id FROM items WHERE id = %s)
                """, (found_id,))
                found_user = cursor.fetchone()

                lost_title = new_details['title'] if type_ == 'lost' else other['title']
                found_title = other['title'] if type_ == 'lost' else new_details['title']
                
                confidence_level, confidence_icon = get_match_confidence_level(final_score)

                # Enhanced notification messages
                message_text_lost = f"""🎯 POTENTIAL MATCH FOUND

{confidence_icon} Match Confidence: {confidence_level} ({round(float(final_score) * 100)}%)

📦 Your Lost Item: {lost_title}
✅ Potentially Found: {found_title}

👤 Found By: {found_user['fullname']}
📧 Email: {found_user['email']}
📱 Phone: {found_user['phone']}

Next Steps:
• Review the match details carefully
• Contact the finder to verify the item
• Arrange a safe meeting location
• Confirm item identity before meeting

⚠️ Safety Reminder: Always meet in public places and verify item details before any exchange."""

                message_text_found = f"""🎯 POTENTIAL MATCH FOUND

{confidence_icon} Match Confidence: {confidence_level} ({round(float(final_score) * 100)}%)

📦 Your Found Item: {found_title}
🔍 Potentially Matches: {lost_title}

👤 Lost By: {lost_user['fullname']}
📧 Email: {lost_user['email']}
📱 Phone: {lost_user['phone']}

Next Steps:
• Review the match details carefully
• Wait for the owner to contact you
• Verify ownership before returning
• Arrange a safe meeting location

⚠️ Safety Reminder: Always meet in public places and ask for proof of ownership before returning items."""

                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, match_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, %s, 'match_found', %s, 0, NOW())
                ''', (lost_user['id'], lost_id, match_id, message_text_lost))

                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, match_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, %s, 'match_found', %s, 0, NOW())
                ''', (found_user['id'], found_id, match_id, message_text_found))
                mysql.connection.commit()

                # Enhanced email
                subject = f"🎯 Reunited Alert: {confidence_level} Confidence Match Found!"
                body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
        .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
        .match-box {{ background: white; padding: 20px; margin: 20px 0; border-left: 4px solid #667eea; border-radius: 5px; }}
        .confidence {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .contact-card {{ background: #fff; border: 1px solid #e0e0e0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .safety-notice {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
        .btn {{ display: inline-block; padding: 12px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 Potential Match Found!</h1>
            <p>Reunited AI has identified a possible match</p>
        </div>
        
        <div class="content">
            <div class="match-box">
                <p class="confidence">{confidence_icon} {confidence_level} Confidence Match</p>
                <p style="font-size: 18px; color: #666;">Match Score: {round(float(final_score) * 100)}%</p>
            </div>
            
            <h2>📦 Item Details</h2>
            <div class="contact-card">
                <p><strong>Lost Item:</strong> {lost_title}</p>
                <p><strong>Found Item:</strong> {found_title}</p>
            </div>
            
            <h2>👥 Contact Information</h2>
            
            <div class="contact-card">
                <h3>🔍 Person Who Lost Item</h3>
                <p><strong>Name:</strong> {lost_user['fullname']}</p>
                <p><strong>Email:</strong> {lost_user['email']}</p>
                <p><strong>Phone:</strong> {lost_user['phone']}</p>
            </div>
            
            <div class="contact-card">
                <h3>✅ Person Who Found Item</h3>
                <p><strong>Name:</strong> {found_user['fullname']}</p>
                <p><strong>Email:</strong> {found_user['email']}</p>
                <p><strong>Phone:</strong> {found_user['phone']}</p>
            </div>
            
            <div class="safety-notice">
                <h3>⚠️ Safety Guidelines</h3>
                <ul>
                    <li>Always meet in well-lit public places</li>
                    <li>Verify item details before meeting</li>
                    <li>Ask for additional proof of ownership</li>
                    <li>Bring a friend if possible</li>
                    <li>Never pay fees upfront</li>
                    <li>Trust your instincts</li>
                </ul>
            </div>
            
            <center>
                <a href="https://reunited.com/dashboard" class="btn">View Match Details</a>
            </center>
            
            <p style="margin-top: 30px; color: #666; font-size: 14px;">
                This is an automated match generated by Reunited's AI system. 
                Please verify all details before proceeding with any exchange.
            </p>
        </div>
    </div>
</body>
</html>
                """

                send_email(mail, subject, [lost_user['email'], found_user['email']], body)
                
                print(f"✅ Match created! Score: {final_score:.2f}, Match ID: {match_id}, Items: {lost_id}↔{found_id}")

        except Exception as e:
            print(f"❌ Error in auto_match for item {other['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue