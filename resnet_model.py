import numpy as np
import re
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

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
# Combined Auto Match (Image + Text)
# -----------------------------
def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql, threshold=0.4):
    """
    Compare new item against opposite type (lost vs found).
    Match based on BOTH image features and text details.
    """

    opposite_type = 'found' if type_ == 'lost' else 'lost'

    cursor.execute('''
        SELECT i.id, i.title, i.description, i.category, i.location_reported, img.ai_features 
        FROM items i
        JOIN images img ON i.id = img.item_id
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

            # ---- Weighted final score ----
            final_score = (0.6 * image_score) + (0.4 * text_score)

            # ---- Insert match if above threshold ----
            if final_score >= threshold:
                cursor.execute('''
                    INSERT INTO ai_matches (lost_item_id, found_item_id, match_score, status)
                    VALUES (%s, %s, %s, 'pending')
                ''', (
                    new_item_id if type_ == 'lost' else other['id'],
                    other['id'] if type_ == 'lost' else new_item_id,
                    round(float(final_score), 2)
                ))
                mysql.connection.commit()

        except Exception as e:
            print(f"Error in auto_match: {e}")
