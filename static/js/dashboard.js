class EmotionDashboard {
  constructor() {
    this.isWebcamActive = false
    this.isMicActive = false
    this.currentEmotion = "neutral"
    this.emotionHistory = []
    this.emotionStats = {} // NEW: Track emotion statistics
    this.mediaStream = null
    this.audioContext = null
    this.analyser = null

    this.emotionConfig = {
      happy: { emoji: "😊", color: "#ffd700", tips: "The person appears joyful and content." },
      sad: { emoji: "😢", color: "#4a90e2", tips: "The person seems to be feeling down or melancholy." },
      angry: { emoji: "😠", color: "#ff6b6b", tips: "The person appears frustrated or upset." },
      fear: { emoji: "😨", color: "#9b59b6", tips: "The person seems anxious or worried." },
      surprise: { emoji: "😲", color: "#f39c12", tips: "The person appears surprised or amazed." },
      disgust: { emoji: "🤢", color: "#27ae60", tips: "The person seems displeased or disgusted." },
      neutral: { emoji: "😐", color: "#95a5a6", tips: "The person appears calm and neutral." },
    }

    this.init()
  }

  init() {
    this.setupEventListeners()
    this.setupThemeToggle()
    this.setupFileUploads()
    this.startEmotionSimulation()
    this.initializeAudioVisualizer()
    this.initializeEmotionStats() // NEW: Initialize emotion statistics
  }

  // NEW: Initialize emotion statistics
  initializeEmotionStats() {
    Object.keys(this.emotionConfig).forEach(emotion => {
      this.emotionStats[emotion] = 0
    })
    this.updateTopEmotions()
  }

  setupEventListeners() {
    // Control buttons
    document.getElementById("webcamBtn").addEventListener("click", () => this.toggleWebcam())
    document.getElementById("micBtn").addEventListener("click", () => this.toggleMicrophone())
    document.getElementById("uploadImageBtn").addEventListener("click", () => this.triggerImageUpload())
    document.getElementById("uploadAudioBtn").addEventListener("click", () => this.triggerAudioUpload())
    document.getElementById("muteBtn").addEventListener("click", () => this.toggleMute())

    // Sensitivity slider
    const sensitivitySlider = document.getElementById("sensitivitySlider")
    sensitivitySlider.addEventListener("input", (e) => {
      document.getElementById("sensitivityValue").textContent = e.target.value
    })

    // Drag and drop
    this.setupDragAndDrop()
  }

  setupThemeToggle() {
    const themeToggle = document.getElementById("themeToggle")
    const currentTheme = localStorage.getItem("theme") || "light"

    if (currentTheme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark")
      themeToggle.innerHTML = '<i class="fas fa-sun"></i>'
    }

    themeToggle.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme")
      const newTheme = currentTheme === "dark" ? "light" : "dark"

      document.documentElement.setAttribute("data-theme", newTheme)
      localStorage.setItem("theme", newTheme)

      themeToggle.innerHTML = newTheme === "dark" ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'

      // Add transition effect
      document.body.style.transition = "all 0.3s ease"
      setTimeout(() => {
        document.body.style.transition = ""
      }, 300)
    })
  }

  async toggleWebcam() {
    const webcamBtn = document.getElementById("webcamBtn")
    const video = document.getElementById("webcamVideo")

    if (!this.isWebcamActive) {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        })
        video.srcObject = this.mediaStream

        webcamBtn.classList.add("active")
        webcamBtn.innerHTML = '<i class="fas fa-video-slash"></i><span>Stop Webcam</span>'
        this.isWebcamActive = true

        this.startFaceDetection()
      } catch (error) {
        console.error("Error accessing webcam:", error)
        this.showNotification("Unable to access webcam", "error")
      }
    } else {
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((track) => track.stop())
      }
      video.srcObject = null

      webcamBtn.classList.remove("active")
      webcamBtn.innerHTML = '<i class="fas fa-video"></i><span>Start Webcam</span>'
      this.isWebcamActive = false
    }
  }

  async toggleMicrophone() {
    const micBtn = document.getElementById("micBtn")

    if (!this.isMicActive) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        this.setupAudioAnalysis(stream)

        micBtn.classList.add("active")
        micBtn.innerHTML = '<i class="fas fa-microphone-slash"></i><span>Stop Microphone</span>'
        this.isMicActive = true

        this.startAudioVisualization()
      } catch (error) {
        console.error("Error accessing microphone:", error)
        this.showNotification("Unable to access microphone", "error")
      }
    } else {
      if (this.audioContext) {
        this.audioContext.close()
      }

      micBtn.classList.remove("active")
      micBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Start Microphone</span>'
      this.isMicActive = false
    }
  }

  setupAudioAnalysis(stream) {
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)()
    this.analyser = this.audioContext.createAnalyser()
    const source = this.audioContext.createMediaStreamSource(stream)

    source.connect(this.analyser)
    this.analyser.fftSize = 256
  }

  startAudioVisualization() {
    const bars = document.querySelectorAll(".audio-bars .bar")
    const bufferLength = this.analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const animate = () => {
      if (!this.isMicActive) return

      this.analyser.getByteFrequencyData(dataArray)

      bars.forEach((bar, index) => {
        const value = dataArray[index * 4] || 0
        const height = (value / 255) * 60 + 20
        bar.style.height = height + "px"
      })

      requestAnimationFrame(animate)
    }

    animate()
  }

  initializeAudioVisualizer() {
    const bars = document.querySelectorAll(".audio-bars .bar")

    // Default animation when no audio input
    setInterval(() => {
      if (!this.isMicActive) {
        bars.forEach((bar, index) => {
          const height = Math.random() * 40 + 20
          bar.style.height = height + "px"
        })
      }
    }, 200)
  }

  startFaceDetection() {
    // Simulate face detection with bounding box
    const canvas = document.getElementById("faceCanvas")
    const ctx = canvas.getContext("2d")
    const video = document.getElementById("webcamVideo")

    const drawFaceBox = () => {
      if (!this.isWebcamActive) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Simulate face detection box
      const boxWidth = 200
      const boxHeight = 250
      const x = (canvas.width - boxWidth) / 2
      const y = (canvas.height - boxHeight) / 2

      ctx.strokeStyle = "#00ff00"
      ctx.lineWidth = 3
      ctx.strokeRect(x, y, boxWidth, boxHeight)

      // Add emotion label
      ctx.fillStyle = "#00ff00"
      ctx.font = "16px Arial"
      ctx.fillText(`${this.currentEmotion.toUpperCase()}`, x, y - 10)

      requestAnimationFrame(drawFaceBox)
    }

    video.addEventListener("loadedmetadata", drawFaceBox)
  }

  setupFileUploads() {
    const imageInput = document.getElementById("imageInput")
    const audioInput = document.getElementById("audioInput")
    const imageUploadArea = document.getElementById("imageUploadArea")
    const audioUploadArea = document.getElementById("audioUploadArea")

    // Image upload
    imageInput.addEventListener("change", (e) => this.handleImageUpload(e.target.files[0]))
    audioInput.addEventListener("change", (e) => this.handleAudioUpload(e.target.files[0]))
  }

  setupDragAndDrop() {
    const uploadAreas = [
      { element: document.getElementById("imageUploadArea"), type: "image" },
      { element: document.getElementById("audioUploadArea"), type: "audio" },
    ]

    uploadAreas.forEach(({ element, type }) => {
      element.addEventListener("dragover", (e) => {
        e.preventDefault()
        element.classList.add("dragover")
      })

      element.addEventListener("dragleave", () => {
        element.classList.remove("dragover")
      })

      element.addEventListener("drop", (e) => {
        e.preventDefault()
        element.classList.remove("dragover")

        const files = e.dataTransfer.files
        if (files.length > 0) {
          if (type === "image") {
            this.handleImageUpload(files[0])
          } else {
            this.handleAudioUpload(files[0])
          }
        }
      })

      element.addEventListener("click", () => {
        if (type === "image") {
          document.getElementById("imageInput").click()
        } else {
          document.getElementById("audioInput").click()
        }
      })
    })
  }

  async handleImageUpload(file) {
    if (!file || !file.type.startsWith("image/")) {
      this.showNotification("Please select a valid image file", "error")
      return
    }

    this.showProcessingModal()

    const formData = new FormData()
    formData.append("image", file)

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()
      this.hideProcessingModal()

      if (result.prediction) {
        this.updateEmotionDisplay(result.prediction)
        this.showImagePreview(file)
      }
    } catch (error) {
      this.hideProcessingModal()
      console.error("Error uploading image:", error)
      this.showNotification("Error processing image", "error")
    }
  }

  async handleAudioUpload(file) {
    if (!file || !file.type.startsWith("audio/")) {
      this.showNotification("Please select a valid audio file", "error")
      return
    }

    this.showProcessingModal()

    const formData = new FormData()
    formData.append("audio", file)

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()
      this.hideProcessingModal()

      if (result.prediction) {
        this.updateEmotionDisplay(result.prediction)
        this.showAudioPreview(file)
      }
    } catch (error) {
      this.hideProcessingModal()
      console.error("Error uploading audio:", error)
      this.showNotification("Error processing audio", "error")
    }
  }

  showImagePreview(file) {
    const preview = document.getElementById("imagePreview")
    const reader = new FileReader()

    reader.onload = (e) => {
      preview.innerHTML = `<img src="${e.target.result}" class="preview-image" alt="Uploaded image">`
      preview.classList.add("show")
    }

    reader.readAsDataURL(file)
  }

  showAudioPreview(file) {
    const preview = document.getElementById("audioPreview")
    const url = URL.createObjectURL(file)

    preview.innerHTML = `<audio controls class="preview-audio"><source src="${url}" type="${file.type}"></audio>`
    preview.classList.add("show")
  }

  updateEmotionDisplay(prediction) {
    let emotion, confidence

    if (Array.isArray(prediction)) {
      // Multiple predictions - take the highest
      emotion = prediction[0][0]
      confidence = prediction[0][1]
    } else {
      emotion = prediction[0]
      confidence = prediction[1]
    }

    this.currentEmotion = emotion.toLowerCase()
    const config = this.emotionConfig[this.currentEmotion]

    // Update main emotion display
    document.getElementById("currentEmoji").textContent = config.emoji
    document.getElementById("emotionName").textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1)
    document.getElementById("confidenceText").textContent = `${confidence.toFixed(1)}%`
    document.getElementById("confidenceFill").style.width = `${confidence}%`
    document.getElementById("emotionTip").textContent = config.tips

    // Update emotion display background
    const emotionDisplay = document.querySelector(".current-emotion")
    emotionDisplay.className = `current-emotion emotion-${this.currentEmotion}`

    // Add to history and update stats
    this.addToEmotionHistory(emotion, confidence)
    this.updateEmotionStats(emotion, confidence) // NEW: Update emotion statistics
  }

  // NEW: Update emotion statistics
  updateEmotionStats(emotion, confidence) {
    const emotionKey = emotion.toLowerCase()
    
    // Update the emotion count with weighted confidence
    if (this.emotionStats[emotionKey] !== undefined) {
      this.emotionStats[emotionKey] += confidence
    }
    
    this.updateTopEmotions()
  }

  // NEW: Update top 3 emotions display
  updateTopEmotions() {
    const topEmotionsGrid = document.getElementById("topEmotionsGrid")
    
    // Sort emotions by confidence score and get top 3
    const sortedEmotions = Object.entries(this.emotionStats)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
    
    topEmotionsGrid.innerHTML = ""
    
    sortedEmotions.forEach(([emotion, score], index) => {
      const config = this.emotionConfig[emotion]
      const percentage = score > 0 ? ((score / Math.max(...Object.values(this.emotionStats))) * 100).toFixed(1) : 0
      
      const emotionCard = document.createElement("div")
      emotionCard.className = "top-emotion-card"
      emotionCard.innerHTML = `
        <div class="emotion-rank">#${index + 1}</div>
        <div class="emotion-emoji">${config.emoji}</div>
        <div class="emotion-name">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</div>
        <div class="emotion-score">${percentage}%</div>
        <div class="emotion-bar">
          <div class="emotion-bar-fill" style="width: ${percentage}%; background-color: ${config.color}"></div>
        </div>
      `
      topEmotionsGrid.appendChild(emotionCard)
    })
  }

  addToEmotionHistory(emotion, confidence) {
    const timestamp = new Date().toLocaleTimeString()
    const historyItem = { emotion, confidence, timestamp }

    this.emotionHistory.unshift(historyItem)
    if (this.emotionHistory.length > 10) {
      this.emotionHistory.pop()
    }

    this.updateEmotionTimeline()
  }

  updateEmotionTimeline() {
    const timeline = document.getElementById("emotionTimeline")
    timeline.innerHTML = ""

    this.emotionHistory.forEach((item) => {
      const config = this.emotionConfig[item.emotion.toLowerCase()]
      const timelineItem = document.createElement("div")
      timelineItem.className = "timeline-item"

      timelineItem.innerHTML = `
                <div class="timeline-emoji">${config.emoji}</div>
                <div class="timeline-info">
                    <div class="timeline-emotion">${item.emotion}</div>
                    <div class="timeline-time">${item.timestamp}</div>
                </div>
                <div class="timeline-confidence">${item.confidence.toFixed(1)}%</div>
            `

      timeline.appendChild(timelineItem)
    })
  }

  startEmotionSimulation() {
    // Simulate random emotion changes for demo
    const emotions = Object.keys(this.emotionConfig)

    setInterval(() => {
      if (!this.isWebcamActive && !this.isMicActive) {
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)]
        const randomConfidence = Math.random() * 40 + 60 // 60-100%

        this.updateEmotionDisplay([randomEmotion, randomConfidence])
      }
    }, 5000)
  }

  triggerImageUpload() {
    document.getElementById("imageInput").click()
  }

  triggerAudioUpload() {
    document.getElementById("audioInput").click()
  }

  toggleMute() {
    const muteBtn = document.getElementById("muteBtn")
    const icon = muteBtn.querySelector("i")

    if (icon.classList.contains("fa-volume-up")) {
      icon.className = "fas fa-volume-mute"
      muteBtn.style.background = "#ff6b6b"
    } else {
      icon.className = "fas fa-volume-up"
      muteBtn.style.background = ""
    }
  }

  showProcessingModal() {
    document.getElementById("processingModal").classList.add("show")
  }

  hideProcessingModal() {
    document.getElementById("processingModal").classList.remove("show")
  }

  showNotification(message, type = "info") {
    const notification = document.createElement("div")
    notification.className = `notification notification-${type}`
    notification.innerHTML = `
            <i class="fas fa-${type === "error" ? "exclamation-triangle" : "info-circle"}"></i>
            <span>${message}</span>
        `

    document.body.appendChild(notification)

    setTimeout(() => {
      notification.classList.add("show")
    }, 100)

    setTimeout(() => {
      notification.classList.remove("show")
      setTimeout(() => notification.remove(), 300)
    }, 3000)
  }
}

// Global logout function
function logout() {
  if (confirm("Are you sure you want to logout?")) {
    window.location.href = "/login"
  }
}

// Initialize dashboard when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new EmotionDashboard()
})

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--card-bg);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: var(--shadow-lg);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        z-index: 1001;
        min-width: 300px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-error {
        border-left: 4px solid var(--error-color);
    }
    
    .notification-success {
        border-left: 4px solid var(--success-color);
    }
    
    .notification-info {
        border-left: 4px solid var(--primary-color);
    }
`

const styleSheet = document.createElement("style")
styleSheet.textContent = notificationStyles
document.head.appendChild(styleSheet)


