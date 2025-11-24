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
# Combined Auto Match (Image + Text) + Notifications + Email
# -----------------------------
def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql, mail, threshold=0.4):
    """
    Compare new item against opposite type (lost vs found).
    Uses enhanced matching with:
    - 60% image similarity (visual matching)
    - 40% text similarity (description + category + location matching)
    
    If match found:
      - Insert into ai_matches
      - Insert notifications for both users
      - Send email alerts
    """
    opposite_type = 'found' if type_ == 'lost' else 'lost'

    # ✅ Fetch opposite items with image features + user details
    # ✅ EXCLUDE items that are already claimed/returned/resolved
    cursor.execute('''
        SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
               img.ai_features,
               CONCAT(u.first_name, ' ', u.last_name) AS fullname,
               u.email, u.phone
        FROM items i
        JOIN images img ON i.id = img.item_id
        JOIN users u ON i.user_id = u.id
        WHERE i.type = %s 
        AND i.status NOT IN ('claimed', 'returned', 'resolved', 'Claimed', 'Returned', 'Resolved')
    ''', (opposite_type,))
    others = cursor.fetchall()

    print(f"🔍 Comparing new {type_} item against {len(others)} active {opposite_type} items")

    for other in others:
        try:
            # ---- Image similarity (60% weight) ----
            image_score = 0.0
            if other['ai_features']:
                other_features = np.array(eval(other['ai_features']))
                image_score = cosine_similarity(new_features, other_features)

            # ---- Enhanced text similarity (40% weight) ----
            other_details = {
                'title': other['title'],
                'description': other['description'],
                'category': other['category'],
                'location': other['location_reported']
            }
            text_score = enhanced_text_similarity(new_details, other_details)

            # ---- Combined score ----
            final_score = (0.6 * image_score) + (0.4 * text_score)

            # Debug logging (optional - comment out in production)
            print(f"Comparing items: {new_details['title']} vs {other['title']}")
            print(f"  Image score: {image_score:.2f}, Text score: {text_score:.2f}, Final: {final_score:.2f}")

            if final_score >= threshold:
                lost_id = new_item_id if type_ == 'lost' else other['id']
                found_id = other['id'] if type_ == 'lost' else new_item_id

                # Check if match already exists to avoid duplicates
                cursor.execute('''
                    SELECT id FROM ai_matches 
                    WHERE (lost_item_id = %s AND found_item_id = %s)
                    OR (lost_item_id = %s AND found_item_id = %s)
                ''', (lost_id, found_id, found_id, lost_id))
                
                existing_match = cursor.fetchone()
                
                if existing_match:
                    print(f"⚠️ Match already exists between items {lost_id} and {found_id}, skipping...")
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
                    SELECT id, CONCAT(first_name, ' ', last_name) AS fullname, email, phone
                    FROM users WHERE id = (SELECT user_id FROM items WHERE id = %s)
                """, (lost_id,))
                lost_user = cursor.fetchone()

                cursor.execute("""
                    SELECT id, CONCAT(first_name, ' ', last_name) AS fullname, email, phone
                    FROM users WHERE id = (SELECT user_id FROM items WHERE id = %s)
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
                
                print(f"✅ Match created! Score: {final_score:.2f}, Match ID: {match_id}")

        except Exception as e:
            print(f"❌ Error in auto_match for item {other['id']}: {e}")
            continue