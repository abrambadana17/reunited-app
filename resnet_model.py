import numpy as np
import re
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from utils import send_email   # ✅ Import helper

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
    
    # Handle case where one or both texts are empty after cleaning
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
    
    Returns: Score between 0 and 1
    """
    
    # 1. Exact category match = bonus points
    category_bonus = 0.0
    if details1.get('category') and details2.get('category'):
        if clean_text(details1['category']) == clean_text(details2['category']):
            category_bonus = 0.2
    
    # 2. Location similarity (important for physical items)
    location_score = 0.0
    loc1 = details1.get('location', '')
    loc2 = details2.get('location', '')
    if loc1 and loc2:
        location_score = text_similarity(loc1, loc2) * 0.2
    
    # 3. Title + Description content matching (main matching)
    content1 = f"{details1.get('title', '')} {details1.get('description', '')}"
    content2 = f"{details2.get('title', '')} {details2.get('description', '')}"
    content_score = text_similarity(content1, content2) * 0.6
    
    # Combine all scores (cap at 1.0)
    final_text_score = min(1.0, category_bonus + location_score + content_score)
    
    return final_text_score

# -----------------------------
# Combined Auto Match with Dynamic Weights
# -----------------------------
def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql, mail, threshold=None):
    """
    Compare new item against opposite type (lost vs found).
    Uses dynamic weights based on whether items have images.
    """
    opposite_type = 'found' if type_ == 'lost' else 'lost'

     # ✅ FIXED: Use LEFT JOIN to include items without images
    if opposite_type == 'lost':
        # When matching against lost items, only include paid lost items
        cursor.execute('''
            SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
                   img.ai_features,
                   CONCAT(u.first_name, ' ', u.last_name) AS fullname,
                   u.email, u.phone
            FROM items i
            LEFT JOIN images img ON i.id = img.item_id  # <-- CHANGED TO LEFT JOIN
            JOIN users u ON i.user_id = u.id
            WHERE i.type = 'lost'
            AND i.status NOT IN ('claimed', 'returned', 'resolved', 'Claimed', 'Returned', 'Resolved')
            AND i.payment_status = 'paid'  -- Only paid lost items
        ''')
    else:
        # For found items, no payment restriction
        cursor.execute('''
            SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
                   img.ai_features,
                   CONCAT(u.first_name, ' ', u.last_name) AS fullname,
                   u.email, u.phone
            FROM items i
            LEFT JOIN images img ON i.id = img.item_id  # <-- CHANGED TO LEFT JOIN
            JOIN users u ON i.user_id = u.id
            WHERE i.type = 'found'
            AND i.status NOT IN ('claimed', 'returned', 'resolved', 'Claimed', 'Returned', 'Resolved')
        ''')

    others = cursor.fetchall()

    print(f"🔍 Comparing new {type_} item {new_item_id} against {len(others)} active {opposite_type} items")
    
    # Determine if new item has image
    new_item_has_image = new_features is not None
    
    # Set threshold based on whether new item has image
    if threshold is None:
        threshold = 0.4 if new_item_has_image else 0.5
    
    for other in others:
        try:
            # ---- Determine weights based on image availability ----
            other_has_image = other['ai_features'] is not None
            
            # Case 1: Both items have images (70% image, 30% text)
            if new_item_has_image and other_has_image:
                image_weight = 0.7
                text_weight = 0.3
            # Case 2: New item has image but other doesn't (use default 60%/40%)
            elif new_item_has_image and not other_has_image:
                image_weight = 0.6
                text_weight = 0.4
            # Case 3: New item has NO image (100% text)
            else:
                image_weight = 0.0
                text_weight = 1.0
            
            print(f"Weights for item {other['id']}: Image={image_weight}, Text={text_weight}")
            
            # ---- Image similarity ----
            image_score = 0.0
            if new_item_has_image and other_has_image:
                try:
                    other_features = np.array(eval(other['ai_features']))
                    image_score = cosine_similarity(new_features, other_features)
                except:
                    image_score = 0.0
            
            # ---- Enhanced text similarity ----
            other_details = {
                'title': other['title'],
                'description': other['description'],
                'category': other['category'],
                'location': other['location_reported']
            }
            text_score = enhanced_text_similarity(new_details, other_details)
            
            # ---- Combined score with dynamic weights ----
            final_score = (image_weight * image_score) + (text_weight * text_score)

            print(f"Comparing: {new_details['title']} (ID:{new_item_id}) vs {other['title']} (ID:{other['id']})")
            print(f"  Image: {image_score:.2f} (weight={image_weight:.1f}), Text: {text_score:.2f} (weight={text_weight:.1f}), Final: {final_score:.2f}, Threshold: {threshold}")

            if final_score >= threshold:
                lost_id = new_item_id if type_ == 'lost' else other['id']
                found_id = other['id'] if type_ == 'lost' else new_item_id

                # ✅ IMPROVED: Better duplicate check
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

                # Insert into ai_matches
                cursor.execute('''
                    INSERT INTO ai_matches (lost_item_id, found_item_id, match_score, status)
                    VALUES (%s, %s, %s, 'pending')
                ''', (lost_id, found_id, round(float(final_score), 2)))
                mysql.connection.commit()

                match_id = cursor.lastrowid 

                # ---- Get both users ----
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

                # ---- Get item titles ----
                lost_title = new_details['title'] if type_ == 'lost' else other['title']
                found_title = other['title'] if type_ == 'lost' else new_details['title']

                # ---- Notifications ----
                message_text_lost = (
                    f"A possible match has been found for your lost item: {lost_title}\n\n"
                    f"Match Score: {round(float(final_score) * 100)}%\n"
                    f"Found Item: {found_title}\n"
                    f"Found By: {found_user['fullname']}\n"
                    f"Email: {found_user['email']}\n"
                    f"Contact: {found_user['phone']}"
                )

                message_text_found = (
                    f"A possible match has been found for your found item: {found_title}\n\n"
                    f"Match Score: {round(float(final_score) * 100)}%\n"
                    f"Lost Item: {lost_title}\n"
                    f"Lost By: {lost_user['fullname']}\n"
                    f"Email: {lost_user['email']}\n"
                    f"Contact: {lost_user['phone']}"
                )

                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, match_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, %s, 'match_found', %s, 0, NOW())
                ''', (lost_user['id'], lost_id, match_id, message_text_lost))

                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, match_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, %s, 'match_found', %s, 0, NOW())
                ''', (found_user['id'], found_id, match_id, message_text_found))
                mysql.connection.commit()

                # ---- Email ----
                subject = "🔔 Reunited: Possible Item Match Found!"
                body = f"""
Hello {lost_user['fullname']} and {found_user['fullname']},

A possible match has been found in Reunited! 🎉

Match Confidence: {round(float(final_score) * 100)}%

Lost Item: {lost_title}
Found Item: {found_title}

For your reference, here are the contact details:

👤 {lost_user['fullname']} (Lost Item)
📧 {lost_user['email']}
📱 {lost_user['phone']}

👤 {found_user['fullname']} (Found Item)
📧 {found_user['email']}
📱 {found_user['phone']}

Please check your Reunited app to view full details.

Best regards,
Reunited Team
                """

                send_email(mail, subject, [lost_user['email'], found_user['email']], body)
                
                print(f"✅ Match created! Score: {final_score:.2f}, Match ID: {match_id}, Items: {lost_id}↔{found_id}")

        except Exception as e:
            print(f"❌ Error in auto_match for item {other['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue