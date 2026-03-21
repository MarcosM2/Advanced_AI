# Imports
import cv2
import time
from ultralytics import YOLO
import threading
import subprocess
import numpy as np
import sounddevice as sd
import whisper

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Load Whisper
whisper_model = whisper.load_model("base")
# Open webcam, typically index 0
cap = cv2.VideoCapture(0)

# Set trigger rules deciding when someone is a real user
MIN_BOX_AREA = 120000 # Minimum boundin box area to consider a person a potential user
HOLD_TIME = 3.0 # Time a potential user must wait before being considered a user
RESET_TIME = 2.0 # Time after the user leaves before system resets

# Whisper audio settings
SAMPLE_RATE = 16000
LISTEN_SECONDS = 4
LISTEN_COOLDOWN = 6.0

# Shared state for listening
lock = threading.Lock()
is_listening = False
talk = False
last_listen_time = 0.0

candidate_time = None # Records time when a potential user was first detected
last_qualified_seen = 0 # Records last time a potential user was detected
greeted = False # Keeps track of greetings so user is only greeted once
hard_reset = False # To reset the system when a user is detected
speech_process = None

def hard_reset_system():
    global greeted, candidate_time, last_qualified_seen, is_listening, talk, hard_reset, speech_process

    with lock:
        hard_reset = True

    sd.stop() # Stop the microphone

    if speech_process is not None:
        speech_process.terminate()

    with lock:
        greeted = False
        candidate_time = None
        last_qualified_seen = 0
        is_listening = False
        talk = False

    print ("[STATE] HARD RESET")


# Windows speech
def _speak_windows(text: str):
    global talk, speech_process, hard_reset
    safe_text = text.replace("'", "''")
    cmd = (
        f"$voice = New-Object -ComObject SAPI.SpVoice; "
        f"$voice.Speak('{safe_text}')"
    )

    with lock:
        if hard_reset:
            return
        talk = True


    speech_process = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    speech_process.wait()

    speech_process = None
    talk = False

# Speech function
def speak(text: str):
    global talk
    with lock:
        if hard_reset:
            return

    print("[SPEAK]", text)
    threading.Thread(target=_speak_windows, args=(text,), daemon=True).start()

# # Whisper listening
# def _listen_and_transcribe():
#     global is_listening, last_listen_time, talk, user_left

#     if user_left == True:
#         user_left = False
#         return

#     while talk == True:
#         time.sleep(0.05)

#     try:
#         print("[WHISPER] Listening...")
#         audio = sd.rec(
#             int(LISTEN_SECONDS * SAMPLE_RATE),
#             samplerate=SAMPLE_RATE,
#             channels=1,
#             dtype="float32"
#         )
#         sd.wait()

#         audio = np.squeeze(audio)

#         print("[WHISPER] Transcribing...")
#         result = whisper_model.transcribe(audio, fp16=False)
#         text = result["text"].strip()

#         if text:
#             print(f"[USER SAID] {text}")
#             talk = True
#             speak(f"You said: {text}. This input will be fed into the LLM for directional advice. Thank you for using PERCI, if you are finished please leave the station and I will reset for the next user. If you have any further quastions please do not hesitate to ask.")
#             _listen_and_transcribe()

#         else:
#             print("[USER SAID] No speech detected.")
#             talk = True
#             speak("Im sorry, I didint quite catch that, could you please repeat the sentence.")
#             _listen_and_transcribe()

#     except Exception as e:
#         print(f"[WHISPER ERROR] {e}")
#     finally:
#         with lock:
#             is_listening = False
#             last_listen_time = time.time()

# Whisper listening
def _listen_and_transcribe():
    global is_listening, last_listen_time
    should_restart = False

    try:
        while True:
            with lock:
                if hard_reset:
                    return
                if not talk:
                    break
            time.sleep(0.05)

        with lock:
            if hard_reset:
                return

        print("[WHISPER] Listening...")
        audio = sd.rec(
            int(LISTEN_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        with lock:
            if hard_reset:
                return

        audio = np.squeeze(audio)

        print("[WHISPER] Transcribing...")
        result = whisper_model.transcribe(audio, fp16=False)
        text = result["text"].strip()

        with lock:
            if hard_reset:
                return

        if text:
            print(f"[USER SAID] {text}")
            speak(
                f"You said: {text}. This input will be fed into the LLM for directional advice. "
                "Thank you for using PERCI. If you are finished, please leave the station and I will reset for the next user. "
                "If you have any further questions, please do not hesitate to ask."
            )
        else:
            print("[USER SAID] No speech detected.")
            speak("I'm sorry, I didn't quite catch that. Could you please repeat the sentence?")

        should_restart = True

    except Exception as e:
        print(f"[WHISPER ERROR] {e}")

    finally:
        with lock:
            is_listening = False
            last_listen_time = time.time()

        if should_restart:
            while True:
                with lock:
                    if hard_reset:
                        return
                    if not talk:
                        break
                time.sleep(0.05)

            start_listening()

def start_listening():
    global is_listening

    with lock:
        if hard_reset:
            return
        if is_listening:
            return
        if time.time() - last_listen_time < LISTEN_COOLDOWN:
            return
        is_listening = True

    threading.Thread(target=_listen_and_transcribe, daemon=True).start()

    
def main():
    global greeted, candidate_time, last_qualified_seen, talk, user_left, hard_reset
    # Main loop, runs continuously until quit
    while True:
        # Read a frame from the camera, ret says if this was successful and frame is the image
        ret, frame = cap.read()
        if not ret:
            print("Camera not working") # Useful for debugging
            break

        # Run YOLO object detection, send frame to model, class 0 is for the person class
        results = model(frame, classes=[0], verbose=False)
        result = results[0] # store detection results for this frame

        # Store the largest bounding box and its area
        largest_box = None
        largest_area = 0

        # Loopp over all detected people
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) # Get coirdinates of bounding box
            area = (x2 - x1) * (y2 - y1) # calculate its area using the coordinates

            # Keep only the largest bounding box, after the loop you will have the biggest detected person in the frame
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        now = time.time() # Get current time in seconds

        # If statement to check if a valid potential user is detected
        if largest_box is not None and largest_area >= MIN_BOX_AREA:
            with lock:
                if hard_reset:
                    hard_reset = False
                    print(" [STATE] READY FOPR NEXT USER")

            last_qualified_seen = now # Store time last potewntial user was seen (updates every frame)

            if candidate_time is None: # Store time potewntial user first appears
                candidate_time = now

            if not greeted and (now - candidate_time >= HOLD_TIME): # If potential user has been present for lonmg enough they are now a user, if system hasent greeted them already do so (Prompt LLM to greet user)
                greeted = True
                talk = True
                candidate_time = None
                speak("Hello I am PERCI, an AI assistant designed to help you find your way around this building. Please tell me, where would you like to go?")
                start_listening()
        else:
            candidate_time = None # If no valid user present simply do nothing

            # If a user was greeted and no user has been seen for long enough reset system to be ready for new user
            if greeted and (now - last_qualified_seen >= RESET_TIME):
                hard_reset_system()

        frame = result.plot() # Draw YOLO detections on the frame

        # If there is a largest bounding box (There will always be one if at least one person is detected) draw the area of the largest box on the frame, this is useful for debugging and understanding how the system works
        if largest_box is not None:
            x1, y1, x2, y2 = largest_box # Extract coordinates of largest box
            # Draw text on fram
            cv2.putText(
                frame, # Draw text on current video frame
                f"Largest area: {largest_area}", # Print largest area and number to screen
                (x1+150, max(30, y1 - 5)), # Where text appears on screen
                cv2.FONT_HERSHEY_SIMPLEX, # Font style
                0.7, # font size
                (0, 0, 0), # Text color
                2 # Text thickness
            )

        # If the user has been greeted and a largest box is detected, draw the area of the largest box on the frame, this is useful for debugging and understanding how the system works
        if greeted == True and largest_box is not None:
            x1, y1, x2, y2 = largest_box # Extract coordinates of largest box
            # Draw green bounding box to highlight user
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3),
            # Draw text on fram
            cv2.putText(
                frame, # Draw text on current video frame
                f"User", # Print largest area and number to screen
                (x1, max(30, y1 + 20)), # Where text appears on screen
                cv2.FONT_HERSHEY_SIMPLEX, # Font style
                0.7, # font size
                (0, 0, 0), # Text color
                2 # Text thickness
            )

        

        cv2.imshow("User Detection", frame) # Open window showing feed

        if cv2.waitKey(1) == ord("q"): # Leave loop if q is pressed
            break

    # Clean up
    sd.stop()

    if speech_process is not None:
        speech_process.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()