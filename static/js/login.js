document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.getElementById("loginForm")
  const usernameInput = document.getElementById("username")
  const passwordInput = document.getElementById("password")
  const errorMessage = document.getElementById("errorMessage")
  const loadingOverlay = document.getElementById("loadingOverlay")
  const loginBtn = document.querySelector(".login-btn")

  // Add input animations
  const inputs = document.querySelectorAll("input")
  inputs.forEach((input) => {
    input.addEventListener("focus", function () {
      this.parentElement.classList.add("focused")
    })

    input.addEventListener("blur", function () {
      if (!this.value) {
        this.parentElement.classList.remove("focused")
      }
    })

    input.addEventListener("input", () => {
      if (errorMessage.classList.contains("show")) {
        errorMessage.classList.remove("show")
      }
    })
  })

  loginForm.addEventListener("submit", (e) => {
    e.preventDefault()

    const username = usernameInput.value.trim()
    const password = passwordInput.value.trim()

    // Show loading state
    loginBtn.classList.add("loading")

    // Simulate authentication delay
    setTimeout(() => {
      if (username === "emotisense" && password === "1234") {
        // Success - show loading overlay and redirect
        loginBtn.classList.remove("loading")
        showLoadingOverlay()

        setTimeout(() => {
          window.location.href = "/dashboard"
        }, 3000)
      } else {
        // Error - show error message
        loginBtn.classList.remove("loading")
        showError()
      }
    }, 1500)
  })

  function showError() {
    errorMessage.classList.add("show")

    // Add shake animation to form
    loginForm.style.animation = "shake 0.5s ease-in-out"

    setTimeout(() => {
      loginForm.style.animation = ""
    }, 500)

    // Hide error after 5 seconds
    setTimeout(() => {
      errorMessage.classList.remove("show")
    }, 5000)
  }

  function showLoadingOverlay() {
    loadingOverlay.classList.add("show")

    // Update loading text dynamically
    const loadingTexts = [
      "Loading emotion recognition models",
      "Initializing neural networks",
      "Calibrating AI systems",
      "Preparing dashboard interface",
      "Almost ready...",
    ]

    const loadingTextElement = document.querySelector(".loading-text")
    let textIndex = 0

    const textInterval = setInterval(() => {
      if (textIndex < loadingTexts.length - 1) {
        textIndex++
        loadingTextElement.textContent = loadingTexts[textIndex]
      } else {
        clearInterval(textInterval)
      }
    }, 600)
  }

  // Add floating shapes animation
  const shapes = document.querySelectorAll(".shape")
  shapes.forEach((shape, index) => {
    shape.addEventListener("mouseenter", function () {
      this.style.animationPlayState = "paused"
      this.style.transform = "scale(1.2)"
    })

    shape.addEventListener("mouseleave", function () {
      this.style.animationPlayState = "running"
      this.style.transform = "scale(1)"
    })
  })

  // Add keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !loginForm.contains(document.activeElement)) {
      usernameInput.focus()
    }
  })

  // Add particle effect on successful login
  function createParticles() {
    const particleContainer = document.createElement("div")
    particleContainer.className = "particle-container"
    document.body.appendChild(particleContainer)

    for (let i = 0; i < 50; i++) {
      const particle = document.createElement("div")
      particle.className = "particle"
      particle.style.left = Math.random() * 100 + "%"
      particle.style.animationDelay = Math.random() * 2 + "s"
      particleContainer.appendChild(particle)
    }

    setTimeout(() => {
      particleContainer.remove()
    }, 3000)
  }

  // Add CSS for particles
  const particleStyles = `
    .particle-container {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 999;
    }
    
    .particle {
      position: absolute;
      width: 4px;
      height: 4px;
      background: linear-gradient(135deg, #667eea, #764ba2);
      border-radius: 50%;
      animation: particleFall 3s linear infinite;
    }
    
    @keyframes particleFall {
      0% {
        transform: translateY(-100vh) rotate(0deg);
        opacity: 1;
      }
      100% {
        transform: translateY(100vh) rotate(360deg);
        opacity: 0;
      }
    }
  `

  const styleSheet = document.createElement("style")
  styleSheet.textContent = particleStyles
  document.head.appendChild(styleSheet)
})
