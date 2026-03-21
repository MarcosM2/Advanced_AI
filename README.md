# Advanced AI - PERCI

PERCI is an AI-powered kiosk assistant that uses:

- YOLO (Ultralytics) for real-time person detection  
- Whisper for speech recognition  
- Text-to-speech (Windows SAPI) for responses  

The system detects when a user approaches, greets them, listens to their request, and responds interactively.

---

## Features

- Real-time person detection using YOLO  
- Voice input using Whisper  
- Text-to-speech responses  
- Automatic user detection and reset system  
- Continuous interaction loop  

---

## Requirements

- Python 3.10 or 3.11 (recommended)  
- Webcam  
- Microphone  
- Windows (for built-in text-to-speech)  
- FFmpeg installed and added to PATH  

---

## Quick Start (Windows)

Use **Command Prompt (cmd)** for easiest setup.

```bash
git clone https://github.com/MarcosM2/Advanced_AI.git
cd Advanced_AI
setup.bat
venv\Scripts\activate
python main.py