import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import re

# Load ResNet50 once
base_model = ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

def extract_features(img_path):
    """Extract vector features from image"""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def text_similarity(str1, str2):
    """Simple Jaccard similarity for text comparison"""
    if not str1 or not str2:
        return 0.0
    words1 = set(re.findall(r'\w+', str1.lower()))
    words2 = set(re.findall(r'\w+', str2.lower()))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)

def auto_match(new_item_id, type_, new_features, new_details, cursor, mysql,
               img_weight=0.7, text_weight=0.3, threshold=0.65):
    """
    Check similarity with opposite type and insert match.
    Uses both image features and textual details.
    """
    opposite_type = 'found' if type_ == 'lost' else 'lost'

    cursor.execute('''
        SELECT i.id, i.title, i.description, img.ai_features
        FROM items i
        JOIN images img ON i.id = img.item_id
        WHERE i.type = %s
    ''', (opposite_type,))
    others = cursor.fetchall()

    for other in others:
        try:
            # Image similarity
            other_features = np.array(eval(other['ai_features']))
            img_score = cosine_similarity(new_features, other_features)

            # Text similarity
            other_details = (other['title'] or '') + " " + (other['description'] or '')
            text_score = text_similarity(new_details, other_details)

            # Weighted final score
            final_score = (img_weight * img_score) + (text_weight * text_score)

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
            print(f"Error matching: {e}")
