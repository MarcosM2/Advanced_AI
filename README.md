# Advanced AI - PERCI

PERCI is an AI-powered kiosk assistant that integrates:

- YOLO (Ultralytics) for real-time user detection  
- Whisper for speech recognition  
- A quantised LLM (Qwen 2.5 via llama-cpp) for natural language understanding  
- A* pathfinding for computing optimal navigation routes  
- Text-to-speech for spoken responses  

The system operates as a full interaction pipeline:

**User detection → Greeting → Speech input → LLM intent parsing → Pathfinding → Spoken directions → Automatic reset**

PERCI is fully asynchronous and designed for real-time interaction.

---

## Features

- Real-time person detection using YOLO  
- Voice input using Whisper  
- Text-to-speech responses  
- Automatic user detection and reset system
- Natural language understanding using an LLM  
- Pathfinding-based navigation using A* 
- Continuous interaction loop 

---

## What PERCI Does

PERCI is an interactive AI kiosk assistant that detects users, listens to spoken requests, and provides real-time navigation guidance.

Example interaction:
1. A user approaches the kiosk → PERCI detects them using a webcam
2. PERCI greets the user automatically
3. The user asks: "Where are the men's toilets?"
4. PERCI processes the request using an LLM
5. It computes the optimal route using A* pathfinding
6. It responds with spoken directions

The system then resets automatically when a user leaves, and gets ready for the next user.
Note: A valid user is considered close enough when their detected bounding box area is at least 120,000 pixels².

---

## Requirements

- Python 3.10 or Newer. Download here: https://www.python.org/downloads/
- Webcam  
- Microphone  
- Windows or MacOS (Tried and tested on windows)
- Git (for cloning the repository). Download here: https://git-scm.com/install/windows
- Microsoft C++ Build Tools. Download here: https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## Quick Start (Windows)

Use **Command Prompt (cmd)** for easiest setup.

```bash
mkdir PERCI
cd PERCI
git clone https://github.com/MarcosM2/Advanced_AI.git
cd Advanced_AI
setup.bat
venv\Scripts\activate
python main.py
```
## Quick Start (macOS)

### Note: Use mainMac.py when running on MacOS

Use a terminal for setup.

```bash
git clone https://github.com/MarcosM2/Advanced_AI.git
cd Advanced_AI
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python mainMac.py
```
