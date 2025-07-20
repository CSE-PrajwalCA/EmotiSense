# EmotiSense: Dual-Modal Emotion Recognition for Autism Support

EmotiSense is a dual-modality emotion recognition system designed to assist individuals with autism, caregivers, and therapists by interpreting emotions from **facial expressions** and **speech tones**. The application leverages deep learning (CNN for face, HuBERT for speech) to analyze uploaded or live-captured data and delivers insights with confidence levels. It is built with a responsive web interface for ease of use in therapy, education, and research.

---

## Overview

Autistic individuals often face difficulties in understanding emotional cues, impacting communication and social interaction. EmotiSense bridges this gap by:

- Detecting facial emotions using a CNN-based deep learning model trained on FER2013.
- Identifying emotions from voice tone using a HuBERT-based speech emotion recognition model trained on RAVDESS.
- Providing a **unified dashboard** for uploading files, viewing predictions, and receiving insights in real-time.

This project was developed as part of **SuPrathon 2k25 Hackathon** by **Team EliteCoders**.

---

## Tech Stack

### Frontend
- **HTML5** – Webpage structure (landing, login, dashboard).
- **CSS3** – Custom, animated, and responsive design.
- **Vanilla JavaScript** – Handles file uploads, drag-and-drop, and API requests.
- **Font Awesome** – For icons and UI indicators.
- **Responsive Design** – Optimized for desktop and mobile.

### Backend
- **Python 3.12** – Primary backend language.
- **Flask (2.3.3)** – REST API for emotion prediction.
- **Flask-CORS (4.0.0)** – Enables frontend-backend integration.
- **TensorFlow (2.16.1)** – Runs emotion recognition models.
- **OpenCV (cv2)** – Preprocesses images (resizing, normalization).
- **PyDub** – Converts audio formats to WAV for analysis.
- **NumPy** – Array and data operations.
- **Scikit-learn (LabelEncoder)** – Decodes emotion predictions for speech.

### Models
- **Facial Emotion Model:** `final_model.keras` – Trained on FER2013 with EfficientNetB0 backbone and autoencoder reconstruction.
- **Speech Emotion Model:** `final_multiinput_model_hubert2.keras` – HuBERT-based classifier trained on RAVDESS.
- **Encoders:** `label_encoder.pkl` and `label_encoder12.pkl` for mapping predictions to emotion labels.

---

## Directory Structure

```

EmotiSense/
│
├── app.py                       
├── requirements.txt            
├── README.md                    
│
├── static/                     
│   ├── css/                   
│   ├── js/                      
│   └── img/                     
│
├── templates/                    
│   ├── index.html              
│   ├── login.html            
│   └── dashboard.html            
│
├── utils/                     
│   ├── helpers.py            
│   └── test.py                  
│   └── models/                      
│         ├── final\_model.keras       
│         ├── final\_multiinput\_model\_hubert2.keras   
│         ├── label\_encoder.pkl    
│         └── label\_encoder12.pkl      
│
├── ipynb codes/                
│   ├── face\_emotion.ipynb       
│   └── speech\_model.ipynb      
│
└── .gitignore                 

```

---

## Features

1. **Dual-Modality Emotion Detection** – Analyzes both facial expressions and speech tone.
2. **Interactive Dashboard** – Simple upload and real-time emotion insights.
3. **Gender & Intensity Customization** – Adapts predictions based on user input.
4. **Autism-Centric Design** – Supports caregivers and therapists by making emotional cues more accessible.

---

## Impact

- Helps autistic individuals, caregivers, and therapists bridge emotional understanding gaps.
- Assists therapists in tracking emotion recognition progress during therapy.
- Provides a platform for researchers studying emotional patterns in speech and expressions.
- Can be scaled for **education**, **mental health apps**, and **social robotics**.

---

## Future Scope

- Build a **mobile app** for cross-platform accessibility.
- Deploy to cloud (AWS/GCP/Azure) for faster, scalable predictions.
- Add **real-time video and live audio capture**.
- Expand models to **support multiple languages and accents**.
- Integrate **gamified emotional learning exercises** for therapy sessions.

---

## Setup Instructions

1. **Clone the repository**
```

git clone <repo-url>
cd EmotiSense

```

2. **Set up a virtual environment**
```

python -m venv venv
venv\Scripts\activate   # On Windows

```

3. **Install dependencies**
```

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

```

4. **Run the app**
```

python app.py

```

5. **Access the app**
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser

```

---

## Team & Acknowledgments

- Developed by **Team EliteCoders** for **SuPrathon 2k25 Hackathon**.
- Models trained using **FER2013** (facial recognition) and **RAVDESS** (speech emotion recognition).
- Special thanks to mentors and open-source contributors.

---
```

---
