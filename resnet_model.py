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
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())

def text_similarity(text1, text2):
    docs = [clean_text(text1), clean_text(text2)]
    vectorizer = TfidfVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()
    return cos_sim([vectors[0]], [vectors[1]])[0][0]

# -----------------------------
# Combined Auto Match (Image + Text) + Notifications + Email
# -----------------------------
def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql, mail, threshold=0.4):
    """
    Compare new item against opposite type (lost vs found).
    If match found:
      - Insert into ai_matches
      - Insert notifications for both users
      - Send email alerts
    """
    opposite_type = 'found' if type_ == 'lost' else 'lost'

    # ✅ Fetch opposite items with image features + user details
    cursor.execute('''
        SELECT i.id, i.user_id, i.title, i.description, i.category, i.location_reported,
               img.ai_features,
               CONCAT(u.first_name, ' ', u.last_name) AS fullname,
               u.email, u.phone
        FROM items i
        JOIN images img ON i.id = img.item_id
        JOIN users u ON i.user_id = u.id
        WHERE i.type = %s
    ''', (opposite_type,))
    others = cursor.fetchall()

    for other in others:
        try:
            # ---- Image similarity ----
            image_score = 0
            if other['ai_features']:
                other_features = np.array(eval(other['ai_features']))
                image_score = cosine_similarity(new_features, other_features)

            # ---- Text similarity ----
            details1 = f"{new_details['title']} {new_details['description']} {new_details['category']} {new_details['location']}"
            details2 = f"{other['title']} {other['description']} {other['category']} {other['location_reported']}"
            text_score = text_similarity(details1, details2)

            final_score = (0.6 * image_score) + (0.4 * text_score)

            if final_score >= threshold:
                lost_id = new_item_id if type_ == 'lost' else other['id']
                found_id = other['id'] if type_ == 'lost' else new_item_id

                # Insert into ai_matches
                cursor.execute('''
                    INSERT INTO ai_matches (lost_item_id, found_item_id, match_score, status)
                    VALUES (%s, %s, %s, 'pending')
                ''', (lost_id, found_id, round(float(final_score), 2)))
                mysql.connection.commit()

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

                # ---- Notifications ----
                message_text_lost = (
                    f"A possible match has been found for your lost item: {new_details['title']}.\n\n"
                    f"Opposite Item: {other['title']}\n"
                    f" Opposite User: {found_user['fullname']}\n"
                    f"Email: {found_user['email']}\n"
                    f"Contact: {found_user['phone']}"
                )

                message_text_found = (
                    f"A possible match has been found for your found item: {other['title']}.\n\n"
                    f"Opposite Item: {new_details['title']}\n"
                    f"Opposite User: {lost_user['fullname']}\n"
                    f"Email: {lost_user['email']}\n"
                    f"Contact: {lost_user['phone']}"
                )


                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, 'match', %s, 0, NOW())
                ''', (lost_user['id'], new_item_id, message_text_lost))

                cursor.execute('''
                    INSERT INTO notifications (user_id, item_id, type, message, is_read, sent_at)
                    VALUES (%s, %s, 'match', %s, 0, NOW())
                ''', (found_user['id'], new_item_id, message_text_found))

                mysql.connection.commit()

                # ---- Email ----
                subject = "🔔 Reunited: Possible Item Match Found!"
                body = f"""
                    Hello {lost_user['fullname']} and {found_user['fullname']},

                    A possible match has been found in Reunited! 🎉

                    Lost Item: {new_details['title']}
                    Found Item: {other['title']}

                    For your reference, here are the contact details:
                    
                    👤 {lost_user['fullname']}  
                    📧 {lost_user['email']}  
                    📱 {lost_user['phone']}

                    👤 {found_user['fullname']}  
                    📧 {found_user['email']}  
                    📱 {found_user['phone']}

                    Please check your Reunited app to confirm and coordinate directly.

                    Best regards,  
                    Reunited Team
                    """

                send_email(mail, subject, [lost_user['email'], found_user['email']], body)

        except Exception as e:
            print(f"❌ Error in auto_match: {e}")
