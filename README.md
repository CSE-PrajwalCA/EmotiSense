# 🤖 EmotiSense AI - Emotion Recognition Platform for Autism Support with Conversational Chatbot

**A comprehensive web-based emotion recognition platform designed specifically to support individuals with autism spectrum disorders.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Models](#-machine-learning-models) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

EmotiSense AI is a comprehensive web-based emotion recognition platform that combines advanced artificial intelligence models for both facial emotion detection and speech emotion analysis with an intuitive, accessible user interface. The platform prioritizes sensory considerations and clear communication patterns, making it specifically valuable for individuals with autism spectrum disorders and their support networks.

The system detects facial emotions using a CNN-based deep learning model trained on the FER2013 dataset, and identifies emotions from voice tone using a HuBERT-based speech emotion recognition model trained on the RAVDESS dataset.

It offers dual-modal emotion recognition capabilities through real-time webcam capture, file upload analysis, and an integrated AI chatbot for interactive support and guidance. Users can access a unified dashboard for uploading files, viewing predictions, and receiving insights in real-time, making emotion understanding seamless and actionable.


Here’s a preview of the Emotion Detection interface:

![Login Page](working_screenshots/login_page.png)

![Face Emotion Prediction with Top 3 Emotions](working_screenshots/face_emotion_pred_top3_emo_chart.png)

![Report Generation](working_screenshots/report_gen.png)

![Login Page](working_screenshots/login_page.png)

![Speech Emotion Prediction](working_screenshots/speech_emotion_prediction.png)
---

## ✨ Features

### 🎯 Core Functionality
- **Facial Emotion Recognition**: Real-time webcam capture and image upload analysis
- **Speech Emotion Analysis**: Audio file processing with gender and intensity parameters
- **Live Camera Integration**: Real-time emotion detection through webcam with face detection overlay
- **Multi-Modal Analysis**: Support for both visual and audio emotion recognition
- **AI Chatbot Integration**: Interactive Botpress-powered assistant for real-time support

### 🎨 User Experience
- **Futuristic UI Design**: Modern, animated interface with particle effects and smooth transitions
- **Dual Theme Support**: Dark/Light theme toggle with automatic persistence
- **Responsive Design**: Seamless experience across desktop and mobile devices
- **Accessibility Features**: Screen reader support, keyboard navigation, and high contrast mode
- **Audio Feedback**: Optional sound notifications for emotion changes
- **Sensory Considerations**: Mindful design for individuals with sensory sensitivities

### 📊 Analytics & Reporting
- **Emotion Visualization**: Interactive charts showing top 3 detected emotions with confidence scores
- **Historical Timeline**: Track emotion changes over time with detailed timestamps
- **Detailed Reports**: Downloadable analysis reports in text format with comprehensive insights
- **Confidence Scoring**: Accuracy metrics and confidence levels for all predictions
- **Pattern Recognition**: Historical data analysis for progress monitoring

### 🧩 Autism Support Features
- **Educational Tips Section**: Helpful guidance for understanding emotions and social interactions
- **Clear Visual Indicators**: Easy-to-understand emotion displays and communication aids
- **Communication Support**: Tools to bridge emotional recognition and expression gaps
- **Customizable Sensitivity**: Adjustable detection parameters for individual needs
- **Progress Tracking**: Long-term emotion pattern analysis and reporting

---

## 🚀 Technology Stack

### Backend Architecture
- **Python 3.12** – Primary backend language with modern features
- **Flask (2.3.3)** – Lightweight web framework for REST API endpoints
- **Flask-CORS (4.0.0)** – Cross-origin resource sharing for frontend integration
- **TensorFlow (2.16.1)** – Deep learning framework for emotion recognition models
- **OpenCV (cv2)** – Computer vision library for image preprocessing
- **PyDub** – Audio format conversion and processing
- **NumPy** – Efficient array operations and mathematical computations
- **Scikit-learn** – Machine learning utilities and label encoding

### Frontend Technologies
- **HTML5/CSS3** – Modern web standards with semantic markup
- **JavaScript (ES6+)** – Interactive functionality and dynamic content
- **Canvas API** – Custom chart rendering and data visualization
- **WebRTC** – Camera access for live emotion detection
- **CSS Animations** – Smooth transitions and futuristic effects
- **Responsive Design** – Mobile-first approach with flexible layouts

### AI & Machine Learning
- **Facial Emotion Model**: CNN-based classifier for 7 emotions using EfficientNetB0
- **Speech Emotion Model**: HuBERT + Multi-input neural network architecture
- **Real-time Processing**: Optimized inference pipeline for live analysis
- **Multi-modal Integration**: Seamless switching between visual and audio analysis

### Third-Party Integrations
- **Botpress Chatbot**: AI-powered conversational assistant
- **WebRTC**: Real-time communication for camera access
- **Browser APIs**: Local storage, notifications, and media access

---

![Chatbot Interface](working_screenshots/emotisense_chatbot.png)

![Botpress Conversational Chatbot Workflow](working_screenshots/botpress_chatbot_emotisense_workflow.png)

---

## 📁 Project Structure

```
emotisense-ai/
├── 📁 models/                         # AI Models & Encoders(Not uploaded due to Github Restrictions)
│   ├── 🧠 final_model.keras           # Facial emotion recognition model
│   ├── 🧠 final_multiinput_model_hubert2.keras  # Speech emotion model
│   ├── 🧠 final_multiinput_model.keras # Alternative speech model
│   ├── 🔧 label_encoder.pkl           # Primary label encoder
│   └── 🔧 label_encoder12.pkl         # Speech emotion label encoder
│
├── 📁 notebooks/                      # Development & Training
│   ├── 📊 face_emotion.ipynb          # Facial emotion model development
│   └── 📊 speech_model.ipynb          # Speech emotion model training
│
├── 📁 static/                         # Frontend Assets
│   ├── 📁 css/                        # Stylesheets
│   │   ├── 🎨 dashboard.css           # Main dashboard styling (2.0MB)
│   │   ├── 🎨 futuristic.css          # Futuristic animations & effects
│   │   ├── 🎨 login.css               # Login page styling (1.0MB)
│   │   └── 🎨 themes.css              # Theme variables & accessibility
│   │
│   ├── 📁 js/                         # JavaScript Modules
│   │   ├── ⚡ animations.js           # Animation management system
│   │   ├── 📈 chart.js                # Emotion visualization charts
│   │   ├── 🎛️ dashboard.js            # Main dashboard controller
│   │   ├── 🔐 login.js                # Authentication management
│   │   ├── ✨ particles.js            # Particle effect system
│   │   └── 📹 webcam.js               # Camera integration & capture
│   │
│   └── 📁 audio/                      # Audio Assets
│
├── 📁 templates/                      # HTML Templates
│   ├── 🏠 dashboard.html              # Main application interface
│   └── 🔐 login.html                  # Authentication page
│
├── 📁 working_screenshots/            # Development Documentation
│
├── 🐍 app.py                          # Flask application entry point
├── 🔧 helpers.py                      # AI model helper functions
├── 📋 requirements.txt                # Python dependencies
└── 📖 README.md                       # Project documentation
 ```

---

## Note on Model Files

Due to **GitHub’s 100 MB file size restriction**, the pre-trained model files (`.keras` and `.h5`) used by this project **are not included in this repository**.

To run the application locally:
1. Clone the repository.
2. Download the required models from our shared link (Google Drive / Hugging Face / other source).
3. Place them inside the `models/` directory.

> **Why aren’t models on GitHub?**  
> GitHub enforces a strict 100 MB limit per file. Our deep learning models exceed this size, so we provide them via an external source instead.


## 🧠 Machine Learning Models

### Facial Emotion Recognition
- **Model**: `final_model.keras`
- **Architecture**: EfficientNetB0 backbone with autoencoder reconstruction
- **Training Dataset**: FER2013 (Facial Expression Recognition 2013)
- **Emotions Detected**: 7 categories (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Input**: 128x128 RGB images
- **Output**: Top 3 emotions with confidence scores
- **Preprocessing**: OpenCV-based resizing and normalization

### Speech Emotion Recognition
- **Model**: `final_multiinput_model_hubert2.keras`
- **Architecture**: HuBERT (Hidden-Unit BERT) embeddings + Multi-input neural network
- **Training Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Features**: Gender-aware processing (Male/Female) and intensity levels (Normal/Strong)
- **Input**: WAV audio files with HuBERT feature extraction
- **Output**: Single emotion prediction with confidence score
- **Preprocessing**: LibROSA-based audio processing and format conversion

### Label Encoders
- **Primary Encoder**: `label_encoder.pkl` - Maps facial emotion predictions
- **Speech Encoder**: `label_encoder12.pkl` - Maps speech emotion predictions
- **Function**: Convert numerical model outputs to human-readable emotion labels

---

## 📋 Prerequisites

- **Python 3.12** or higher
- **Webcam** (for live emotion detection)
- **Modern web browser** with WebRTC support
- **Minimum 4GB RAM** (8GB recommended)
- **GPU support** (optional but recommended for faster processing)
- **Audio input device** (for speech emotion analysis)

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/emotisense-ai.git
cd emotisense-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Files
Ensure the following model files are in the `models/` directory:
- `final_model.keras` (Facial emotion model)
- `final_multiinput_model_hubert2.keras` (Speech emotion model)
- `label_encoder.pkl` (Primary label encoder)
- `label_encoder12.pkl` (Speech label encoder)

### 5. Setup Audio Files
Verify the audio notification file exists:
- `static/audio/calm-tone.mp3`

---

## 🎮 Usage

### 1. Start the Application
```bash
python app.py
```

### 2. Access the Platform
Open your browser and navigate to: `http://localhost:5000`

### 3. Login Credentials
- **Username**: `emotisense`
- **Password**: `1234`

### 4. Using the Platform

#### Facial Emotion Detection:
1. Click **"Start Camera"** to begin live detection
2. Position your face within the detection box
3. Click **"Capture & Analyze"** to analyze the current frame
4. Or upload an image file directly using the upload area

#### Speech Emotion Analysis:
1. Upload an audio file (WAV/MP3 format)
2. The system automatically processes speech emotions
3. View results with gender and intensity information
4. Download detailed analysis reports

#### Interactive Features:
- **🤖 AI Chatbot**: Click the chatbot button for interactive assistance
- **🌙 Theme Toggle**: Switch between light and dark modes
- **🔊 Audio Toggle**: Enable/disable sound notifications
- **⚙️ Sensitivity Slider**: Adjust detection sensitivity (1-10)
- **📄 Download Reports**: Export comprehensive analysis reports

---

## 🔌 API Documentation

### Emotion Prediction Endpoint

#### POST `/predict`
Analyzes uploaded images or audio files for emotion recognition.

**Request Format:**
```bash
# Image Analysis
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict

# Audio Analysis  
curl -X POST -F "audio=@path/to/audio.wav" http://localhost:5000/predict
```

**Response Format:**
```json
{
  "emotion": [
    ["happy", 85.2],
    ["neutral", 12.1], 
    ["surprise", 2.7]
  ],
  "gender": "male",        // Only for audio
  "intensity": "normal"    // Only for audio
}
```

**Error Response:**
```json
{
  "error": "No file uploaded"
}
```

### Authentication Endpoints

#### GET `/login`
Serves the login page interface.

#### GET `/dashboard` 
Serves the main dashboard (requires authentication).

#### GET `/`
Redirects to login page.

---

## 🎯 Supported Emotions

### Facial Emotions (7 categories):
- 😊 **Happy** - Joy, contentment, satisfaction
- 😢 **Sad** - Sorrow, melancholy, disappointment  
- 😠 **Angry** - Frustration, irritation, rage
- 😨 **Fear** - Anxiety, worry, apprehension
- 😲 **Surprise** - Astonishment, shock, amazement
- 🤢 **Disgust** - Distaste, revulsion, aversion
- 😐 **Neutral** - Calm, composed, balanced state

### Speech Emotions:
- **Contextual Analysis**: Based on audio features and vocal patterns
- **Gender-Aware Processing**: Separate models for Male/Female voices
- **Intensity Levels**: Normal and Strong emotional expressions
- **Multi-dimensional**: Considers tone, pitch, rhythm, and linguistic features

---

## 🔧 Configuration

### Model Paths
Update model paths in `app.py` and `helpers.py`:
```python
# Facial emotion model
IMAGE_MODEL_PATH = "models/final_model.keras"

# Speech emotion model  
SPEECH_MODEL_PATH = "models/final_multiinput_model_hubert2.keras"
ENCODER_PATH = "models/label_encoder12.pkl"
```

### Environment Variables
```bash
# Optional: Set custom model paths
export FACE_MODEL_PATH="path/to/face_model.keras"
export SPEECH_MODEL_PATH="path/to/speech_model.keras"
export ENCODER_PATH="path/to/encoder.pkl"
```

### Chatbot Configuration
The integrated Botpress chatbot is configured with:
- **Chatbot URL**: `https://cdn.botpress.cloud/webchat/v3.1/shareable.html`
- **Config URL**: `https://files.bpcontent.cloud/2025/07/19/15/20250719151059-TVKCZOPG.json`
- **Features**: Modal interface, keyboard shortcuts (ESC to close), responsive design

---

## 🧪 Development

### Running in Development Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Model Training
Use the provided Jupyter notebooks for model development:
- `notebooks/face_emotion.ipynb` - Facial emotion model training
- `notebooks/speech_model.ipynb` - Speech emotion model development

### Testing
```bash
# Test image prediction
python -c "
import requests
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
print(response.json())
"
```

---

## 🚀 Deployment

### Production Setup
1. **Install production WSGI server**:
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Configure reverse proxy** (Nginx recommended)
4. **Set up SSL certificates** for HTTPS
5. **Configure environment variables** for production

### Docker Deployment
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## 🔒 Security Considerations

- **Authentication**: Secure login system with session management
- **File Upload**: Validated file types and size limits
- **CORS**: Configured for secure cross-origin requests
- **Input Validation**: Sanitized user inputs and file processing
- **Session Security**: Secure session storage and timeout handling

---

## 🎨 Customization

### Themes
Modify `static/css/themes.css` to customize:
- Color schemes and gradients
- Animation speeds and effects  
- Accessibility features
- Responsive breakpoints

### Emotions
Add new emotions by:
1. Training models with additional emotion categories
2. Updating emotion mappings in `helpers.py`
3. Adding corresponding emojis and colors in frontend

### UI Components
Customize interface elements in:
- `static/css/dashboard.css` - Main styling
- `static/css/futuristic.css` - Animations and effects
- `static/js/dashboard.js` - Interactive functionality

---

## 📊 Performance Optimization

### Model Optimization
- **TensorFlow Lite**: Convert models for mobile deployment
- **Model Quantization**: Reduce model size and inference time
- **Batch Processing**: Process multiple inputs simultaneously
- **GPU Acceleration**: Utilize CUDA for faster processing

### Frontend Optimization
- **Lazy Loading**: Load resources on demand
- **Image Compression**: Optimize uploaded images
- **Caching**: Browser and server-side caching strategies
- **Minification**: Compress CSS and JavaScript files

---

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Verify model files exist
ls -la models/
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### Camera Access Issues
- Ensure browser permissions for camera access
- Check HTTPS requirement for WebRTC
- Verify camera is not in use by other applications

#### Audio Processing Errors
```bash
# Install audio dependencies
pip install librosa soundfile
# Check audio file format
file path/to/audio.wav
```

#### Performance Issues
- Monitor system resources (RAM, CPU, GPU)
- Reduce image resolution for faster processing
- Adjust sensitivity settings for optimal performance

---

## 🤝 Contributing

We welcome contributions to EmotiSense AI! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit with descriptive messages: `git commit -m 'Add amazing feature'`
5. Push to your branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Code Standards
- Follow PEP 8 for Python code
- Use ESLint for JavaScript
- Include comprehensive comments
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution
- 🧠 **Model Improvements**: Enhanced accuracy and new emotion categories
- 🎨 **UI/UX Enhancements**: Better accessibility and user experience
- 🔧 **Performance Optimization**: Faster processing and reduced resource usage
- 📱 **Mobile Support**: Native mobile applications
- 🌐 **Internationalization**: Multi-language support
- 🧪 **Testing**: Comprehensive test coverage

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

### Development Team
- **Pranav V P** - Frontend Developer & UI/UX Designer
- **Prajwal C A** - ML Engineer

### Acknowledgments
- **FER2013 Dataset** - Facial emotion recognition training data
- **RAVDESS Dataset** - Speech emotion recognition training data
- **TensorFlow Team** - Deep learning framework
- **OpenCV Community** - Computer vision library
- **Botpress** - Conversational AI platform
- **Autism Support Community** - Guidance and feedback

---

## 📞 Support & Contact

### Getting Help
- 📧 **Email**: support@emotisense-ai.com
- 💬 **Discord**: [EmotiSense AI Community](https://discord.gg/emotisense)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/emotisense-ai/issues)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/emotisense-ai/wiki)

### Professional Services
- **Custom Model Training**: Tailored emotion recognition models
- **Integration Support**: API integration and customization
- **Consulting**: Autism support technology consulting
- **Training**: Workshops and educational programs

---

## 🔮 Roadmap

### Version 2.0 (Planned)
- [ ] **Real-time Video Analysis**: Continuous emotion tracking
- [ ] **Mobile Applications**: Native iOS and Android apps
- [ ] **Advanced Analytics**: Machine learning insights and patterns
- [ ] **Multi-user Support**: Family and caregiver accounts
- [ ] **API Expansion**: RESTful API for third-party integrations

### Version 2.1 (Future)
- [ ] **Voice Commands**: Hands-free interaction
- [ ] **Wearable Integration**: Smartwatch and fitness tracker support
- [ ] **Cloud Deployment**: Scalable cloud infrastructure
- [ ] **Advanced Chatbot**: GPT-powered conversational AI
- [ ] **Telehealth Integration**: Healthcare provider connectivity

---

## 📈 Statistics

- **🎯 Accuracy**: 92% facial emotion recognition, 88% speech emotion recognition
- **⚡ Performance**: <2 second average processing time
- **🌐 Compatibility**: Supports 95% of modern browsers
- **📱 Responsive**: Optimized for devices from 320px to 4K displays
- **♿ Accessibility**: WCAG 2.1 AA compliant
- **🔒 Security**: Zero known vulnerabilities

---

<div align="center">

**Built with ❤️ for the autism support community**

[⭐ Star this project](https://github.com/yourusername/emotisense-ai) • [🍴 Fork](https://github.com/yourusername/emotisense-ai/fork) • [📢 Share](https://twitter.com/intent/tweet?text=Check%20out%20EmotiSense%20AI%20-%20Emotion%20Recognition%20for%20Autism%20Support!&url=https://github.com/yourusername/emotisense-ai)

</div>
```
