# # utils/helpers.py

# import numpy as np
# import joblib
# import os

# # === Load label encoder (for speech model) ===
# ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
# try:
#     le_emotion = joblib.load(ENCODER_PATH)
#     print("[INFO] Speech label encoder loaded.")
# except Exception as e:
#     print(f"[WARN] Could not load speech label encoder: {e}")
#     le_emotion = None

# # === 1. For Speech Model ===
# def decode_prediction(pred):
#     """
#     Decodes the output of the speech model using LabelEncoder.
#     If confidence < 55%, returns sorted class probabilities.
#     """
#     if len(pred.shape) == 1:
#         pred = np.expand_dims(pred, axis=0)

#     conf = float(np.max(pred)) * 100
#     label_index = int(np.argmax(pred))

#     if le_emotion:
#         class_names = le_emotion.inverse_transform(np.arange(pred.shape[1]))
#     else:
#         class_indices = {
#             'angry': 0, 'disgust': 1, 'fear': 2,
#             'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
#         }
#         index_to_class = {v: k for k, v in class_indices.items()}
#         class_names = [index_to_class[i] for i in range(pred.shape[1])]

#     if conf < 55:
#         sorted_indices = np.argsort(pred[0])[::-1]
#         return [(class_names[i], float(pred[0][i]) * 100) for i in sorted_indices]

#     return class_names[label_index], conf

# # === 2. For Face Image Model ===
# def decode_face_output(pred):
#     """
#     Decodes output from the face image model (EfficientNet).
#     Returns the top predicted label and confidence.
#     If confidence < 55%, returns sorted class probabilities.
#     """
#     face_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#     if len(pred.shape) == 1:
#         pred = np.expand_dims(pred, axis=0)

#     conf = float(np.max(pred)) * 100
#     label_index = int(np.argmax(pred))

#     if conf < 55:
#         sorted_indices = np.argsort(pred[0])[::-1]
#         return [(face_classes[i], float(pred[0][i]) * 100) for i in sorted_indices]

#     return face_classes[label_index], conf
# utils/helpers.py



# === 2. For Face Image Model ===
# def decode_face_output(pred):
#     """
#     Decodes output from the face image model (EfficientNet).
#     Returns the top predicted label and confidence.
#     If confidence < 55%, returns sorted class probabilities.
#     """
#     face_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#     # Convert to NumPy array if it's a list
#     pred = np.array(pred)

#     if len(pred.shape) == 1:
#         pred = np.expand_dims(pred, axis=0)

#     conf = float(np.max(pred)) * 100
#     label_index = int(np.argmax(pred))

#     if conf < 55:
#         sorted_indices = np.argsort(pred[0])[::-1]
#         return [(face_classes[i], float(pred[0][i]) * 100) for i in sorted_indices]

 #   return face_classes[label_index], conf
# def decode_face_output(pred):
#     pred = np.array(pred)
#     face_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#     pred = np.array(pred)  # ensures NumPy format

#     if len(pred.shape) == 1:
#         pred = np.expand_dims(pred, axis=0)  # ensures (1, 7)

#     conf = float(np.max(pred)) * 100
#     label_index = int(np.argmax(pred))

#     if conf < 55:
#         sorted_indices = np.argsort(pred[0])[::-1]
#         return [(face_classes[i], float(pred[0][i]) * 100) for i in sorted_indices]

#     return face_classes[label_index], conf




























# import numpy as np
# import joblib
# import os
# import tensorflow as tf
# # === Load label encoder (for speech model only) ===
# ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
# try:
#     le_emotion = joblib.load(ENCODER_PATH)
#     print("[INFO] Speech label encoder loaded.")
# except Exception as e:
#     print(f"[WARN] Could not load speech label encoder: {e}")
#     le_emotion = None

# # === 1. For Speech Model ===
# def decode_prediction(pred):
#     """
#     Decodes the output of the speech model using LabelEncoder.
#     If confidence < 55%, returns sorted class probabilities.
#     """
#     if len(pred.shape) == 1:
#         pred = np.expand_dims(pred, axis=0)

#     conf = float(np.max(pred)) * 100
#     label_index = int(np.argmax(pred))

#     # If LabelEncoder is available, use it
#     if le_emotion:
#         try:
#             class_names = le_emotion.classes_
#         except AttributeError:
#             # Fallback: regenerate class names from inverse transform
#             class_names = le_emotion.inverse_transform(np.arange(pred.shape[1]))
#     else:
#         # Fallback hardcoded labels if encoder missing
#         class_indices = {
#             'angry': 0, 'disgust': 1, 'fear': 2,
#             'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
#         }
#         index_to_class = {v: k for k, v in class_indices.items()}
#         class_names = [index_to_class[i] for i in range(pred.shape[1])]

#     if conf < 55:
#         sorted_indices = np.argsort(pred[0])[::-1]
#         return [(class_names[i], float(pred[0][i]) * 100) for i in sorted_indices]

#     return class_names[label_index], conf


# def decode_face_output(pred):
#     """
#     Decodes class probabilities from the dual-output model (emotion classification + autoencoder).
    
#     Args:
#         pred (list or tuple): [class_probs, image_reconstruction]

#     Returns:
#         str or list: Most probable emotion label and confidence,
#                      or top-3 predictions if confidence is low.
#     """
#     face_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

#     # Extract the class prediction (first element)
#     class_probs = pred[0]  # shape (1, 7)
    
#     if isinstance(class_probs, tf.Tensor):
#         class_probs = class_probs.numpy()

#     if len(class_probs.shape) == 1:
#         class_probs = np.expand_dims(class_probs, axis=0)

#     conf = float(np.max(class_probs)) * 100
#     label_index = int(np.argmax(class_probs))

#     if conf < 55:
#         sorted_indices = np.argsort(class_probs[0])[::-1]
#         return [(face_classes[i], float(class_probs[0][i]) * 100) for i in sorted_indices[:3]]

#     return face_classes[label_index], conf
















# utils/helpers.py

import numpy as np
import joblib
import os
import tensorflow as tf
# === Load label encoder (for speech model only) ===
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
try:
    le_emotion = joblib.load(ENCODER_PATH)
    print("[INFO] Speech label encoder loaded.")
except Exception as e:
    print(f"[WARN] Could not load speech label encoder: {e}")
    le_emotion = None

# === 1. For Speech Model ===
def decode_prediction(pred):
    """
    Decodes the output of the speech model using LabelEncoder.
    If confidence < 55%, returns sorted class probabilities.
    """
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)

    conf = float(np.max(pred)) * 100
    label_index = int(np.argmax(pred))

    # If LabelEncoder is available, use it
    if le_emotion:
        try:
            class_names = le_emotion.classes_
        except AttributeError:
            # Fallback: regenerate class names from inverse transform
            class_names = le_emotion.inverse_transform(np.arange(pred.shape[1]))
    else:
        # Fallback hardcoded labels if encoder missing
        class_indices = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
        }
        index_to_class = {v: k for k, v in class_indices.items()}
        class_names = [index_to_class[i] for i in range(pred.shape[1])]

    if conf < 55:
        sorted_indices = np.argsort(pred[0])[::-1]
        return [(class_names[i], float(pred[0][i]) * 100) for i in sorted_indices]

    return class_names[label_index], conf

# === 2. For Face Image Model ===
def decode_face_output(pred):
    """
    Decodes class probabilities from the dual-output model (emotion classification + autoencoder).
    
    Args:
        pred (list or tuple): [class_probs, image_reconstruction]

    Returns:
        str or list: Most probable emotion label and confidence,
                     or top-3 predictions if confidence is low.
    """
    face_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Extract the class prediction (first element)
    class_probs = pred[0]  # shape (1, 7)
    
    if isinstance(class_probs, tf.Tensor):
        class_probs = class_probs.numpy()

    if len(class_probs.shape) == 1:
        class_probs = np.expand_dims(class_probs, axis=0)

    conf = float(np.max(class_probs)) * 100
    label_index = int(np.argmax(class_probs))

    if conf < 55:
        sorted_indices = np.argsort(class_probs[0])[::-1]
        return [(face_classes[i], float(class_probs[0][i]) * 100) for i in sorted_indices[:3]]

    return face_classes[label_index], conf
