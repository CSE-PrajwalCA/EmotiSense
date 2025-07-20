# # app.py
# from flask import Flask, render_template, request, jsonify
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import os
# from utils.helpers import decode_prediction
# from pydub import AudioSegment
# import tempfile
# from utils.helpers import decode_prediction, decode_face_output

# app = Flask(__name__)
# IMG_SIZE = 128

# # Load models once
# image_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\emotion_app\emotion_app\models\final_model.keras')
# speech_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\emotion_app\emotion_app\models\final_multiinput_model.keras')

# def preprocess_image(image_file):
#     image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     image = image.astype('float32') / 255.0
#     return np.expand_dims(image, axis=0)

# def preprocess_audio(audio_file):
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
#         audio = AudioSegment.from_file(audio_file)
#         audio.export(tmp.name, format='wav')
#         # Dummy audio preprocessing for demonstration
#         waveform = np.random.rand(1, 128, 128, 1)
#         return waveform

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' in request.files:
#         img = request.files['image']
#         img_data = preprocess_image(img)
#         pred = image_model.predict(img_data)
#         print("         🥰 mean?             ",pred)
#         result = decode_face_output(pred)  # ← Use image decoder here

#     elif 'audio' in request.files:
#         audio = request.files['audio']
#         audio_data = preprocess_audio(audio)
#         pred = speech_model.predict(audio_data)
#         result = decode_prediction(pred)  # ← Use speech decoder here

#     else:
#         return jsonify({'error': 'No file uploaded'})

#     return jsonify({'prediction': result})


# if __name__ == '__main__':
#     app.run(debug=True)








from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from utils.helpers import decode_prediction
from pydub import AudioSegment
import tempfile
from utils.helpers import decode_prediction, decode_face_output

app = Flask(__name__)
IMG_SIZE = 128

# Load models once
image_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_model.keras')
speech_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_multiinput_model.keras')

def preprocess_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_audio(audio_file):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio = AudioSegment.from_file(audio_file)
        audio.export(tmp.name, format='wav')
        # Dummy audio preprocessing for demonstration
        waveform = np.random.rand(1, 128, 128, 1)
        return waveform

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        img = request.files['image']
        img_data = preprocess_image(img)
        pred = image_model.predict(img_data)
        print("         🥰 mean?             ",pred)
        result = decode_face_output(pred)  # ← Use image decoder here

    elif 'audio' in request.files:
        audio = request.files['audio']
        audio_data = preprocess_audio(audio)
        pred = speech_model.predict(audio_data)
        result = decode_prediction(pred)  # ← Use speech decoder here

    else:
        return jsonify({'error': 'No file uploaded'})

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)















# from flask import Flask, render_template, request, jsonify
# from tensorflow.keras.models import load_model
# import numpy as np
# import cv2
# import os
# from utils.helpers import decode_prediction
# from pydub import AudioSegment
# import tempfile
# from utils.helpers import decode_prediction, decode_face_output
# import base64
# from io import BytesIO
# from PIL import Image

# app = Flask(__name__)
# IMG_SIZE = 128

# # Load models once
# try:
#     image_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_model.keras')
#     speech_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_multiinput_model.keras')
#     print("Models loaded successfully!")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     image_model = None
#     speech_model = None

# def preprocess_image(image_file):
#     """Preprocess image for emotion recognition"""
#     try:
#         # Handle different input types
#         if hasattr(image_file, 'read'):
#             # File upload
#             image_data = image_file.read()
#             image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#         else:
#             # Base64 or other formats
#             image = cv2.imdecode(np.frombuffer(image_file, np.uint8), cv2.IMREAD_COLOR)
        
#         if image is None:
#             raise ValueError("Could not decode image")
            
#         # Resize and normalize
#         image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#         image = image.astype('float32') / 255.0
        
#         return np.expand_dims(image, axis=0)
#     except Exception as e:
#         print(f"Error preprocessing image: {e}")
#         return None

# def preprocess_audio(audio_file):
#     """Preprocess audio for emotion recognition"""
#     try:
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
#             audio = AudioSegment.from_file(audio_file)
#             audio.export(tmp.name, format='wav')
            
#             # For now, return dummy data - replace with actual audio preprocessing
#             # This should be replaced with proper audio feature extraction
#             waveform = np.random.rand(1, 128, 128, 1)
#             return waveform
#     except Exception as e:
#         print(f"Error preprocessing audio: {e}")
#         return None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle emotion prediction requests"""
#     try:
#         if 'image' in request.files:
#             img = request.files['image']
#             if img.filename == '':
#                 return jsonify({'error': 'No image selected'})
            
#             if image_model is None:
#                 return jsonify({'error': 'Image model not loaded'})
            
#             img_data = preprocess_image(img)
#             if img_data is None:
#                 return jsonify({'error': 'Error processing image'})
            
#             pred = image_model.predict(img_data)
#             print("Image prediction:", pred)
#             result = decode_face_output(pred)
            
#             return jsonify({
#                 'prediction': result,
#                 'type': 'image',
#                 'confidence': float(np.max(pred)) * 100
#             })

#         elif 'audio' in request.files:
#             audio = request.files['audio']
#             if audio.filename == '':
#                 return jsonify({'error': 'No audio selected'})
            
#             if speech_model is None:
#                 return jsonify({'error': 'Speech model not loaded'})
            
#             audio_data = preprocess_audio(audio)
#             if audio_data is None:
#                 return jsonify({'error': 'Error processing audio'})
            
#             pred = speech_model.predict(audio_data)
#             print("Audio prediction:", pred)
#             result = decode_prediction(pred)
            
#             return jsonify({
#                 'prediction': result,
#                 'type': 'audio',
#                 'confidence': float(np.max(pred)) * 100
#             })

#         else:
#             return jsonify({'error': 'No file uploaded'})

#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return jsonify({'error': f'Prediction failed: {str(e)}'})

# @app.route('/predict_webcam', methods=['POST'])
# def predict_webcam():
#     """Handle webcam capture prediction"""
#     try:
#         data = request.get_json()
#         if 'image' not in data:
#             return jsonify({'error': 'No image data provided'})
        
#         # Decode base64 image
#         image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
#         image_bytes = base64.b64decode(image_data)
        
#         if image_model is None:
#             return jsonify({'error': 'Image model not loaded'})
        
#         img_data = preprocess_image(image_bytes)
#         if img_data is None:
#             return jsonify({'error': 'Error processing webcam image'})
        
#         pred = image_model.predict(img_data)
#         result = decode_face_output(pred)
        
#         return jsonify({
#             'prediction': result,
#             'type': 'webcam',
#             'confidence': float(np.max(pred)) * 100
#         })
        
#     except Exception as e:
#         print(f"Webcam prediction error: {e}")
#         return jsonify({'error': f'Webcam prediction failed: {str(e)}'})

# @app.errorhandler(404)
# def not_found(error):
#     return render_template('404.html'), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return render_template('500.html'), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
