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
# image_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_model.keras')
# speech_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_multiinput_model.keras')

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

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html')

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
import tempfile
from pydub import AudioSegment

# Import helper functions
from utils.helpers import decode_face_output, predict_emotion_from_audio

app = Flask(__name__)
IMG_SIZE = 128

# Load facial emotion model only
image_model = load_model(r'C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_model.keras')

# HuBERT model paths (for audio emotion recognition)
HUBERT_MODEL_PATH = r"C:\Users\www12\OneDrive\Documents\Projects\em2\models\final_multiinput_model_hubert2.keras"
HUBERT_ENCODER_PATH = r"C:\Users\www12\OneDrive\Documents\Projects\em2\models\label_encoder12.pkl"


def preprocess_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)


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
    # Handle image prediction (Facial Emotion Recognition)
    if 'image' in request.files:
        img = request.files['image']
        img_data = preprocess_image(img)
        pred = image_model.predict(img_data)
        result = decode_face_output(pred)

    # Handle audio prediction (HuBERT-based Emotion Recognition)
    elif 'audio' in request.files:
        audio = request.files['audio']

        # Save uploaded audio as a temp WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.save(tmp.name)
            wav_path = tmp.name

        # Get gender and intensity (optional form params)
        gender = request.form.get('gender', 'male')
        intensity = request.form.get('intensity', 'normal')

        # Predict using HuBERT-based logic
        emotion, confidence = predict_emotion_from_audio(
            wav_path, HUBERT_MODEL_PATH, HUBERT_ENCODER_PATH, gender, intensity
        )

        result = {'emotion': emotion, 'confidence': float(confidence)}

    else:
        return jsonify({'error': 'No file uploaded'})

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)












