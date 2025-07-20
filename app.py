# app.py
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
image_model = load_model(r'C:\Project\Hackathon\emotion_app\emotion_app\models\final_model.keras')
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
    return render_template('index (4).html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = {}

        if 'image' in request.files:
            img = request.files['image']
            img_data = preprocess_image(img)
            pred = image_model.predict(img_data)
            print("         🥰 mean?             ",pred)
            result = decode_face_output(pred)  # ← Use image decoder here
        elif 'audio' in request.files and request.files['audio'].filename != '':
            audio = request.files['audio']

            # Save uploaded audio to temp wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio.save(tmp.name)
                wav_path = tmp.name

            # Default fallback values
            gender = request.form.get('gender', 'male')
            intensity = request.form.get('intensity', 'normal')

            from utils.helpers import predict_emotion_from_audio
            model_path = r"C:\Project\Hackathon\emotion_app\emotion_app\models\final_multiinput_model_hubert2.keras"
            encoder_path = r"C:\Project\Hackathon\emotion_app\emotion_app\models\label_encoder12.pkl"

            emotion, confidence = predict_emotion_from_audio(
                wav_path, model_path, encoder_path, gender, intensity
            )
            result['emotion'] = (emotion, confidence)
            result['gender'] = gender
            result['intensity'] = intensity

        else:
            return jsonify({'error': 'No file uploaded'}), 400

        return jsonify(result)

    except Exception as e:
        print(f"🔥 Prediction error: {e}")
        return jsonify({'error': 'Prediction failed. Try again.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
