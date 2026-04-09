# Imports
import cv2
import time
from ultralytics import YOLO
import threading
import subprocess
import numpy as np
import sounddevice as sd
import whisper
import sys
import json
import pyttsx3
from collections import defaultdict


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
SAMPLE_RATE = 16000 # Microphone sample rate
LISTEN_SECONDS = 4 # Whisper listening window
LISTEN_COOLDOWN = 6.0 # minimum wait before listening again


lock = threading.Lock() # General shared lock state
is_listening = False # Is a listening thread currently active?
talk = False # is speech curently active?
last_listen_time = 0.0 # timestamp of last listening cycle
listen_thread = None # store current listening thread object
session_id = 0 # Identify current session
reset_in_progress = False # prevents overlapping resets
pending_speeches = defaultdict(int) # counts how many speech tasks are still outstanding for a current session
llm_lock = threading.Lock() # Guards LLM process access
speech_lock = threading.Lock() # Only one TTS at a time

candidate_time = None # Records time when a potential user was first detected
last_qualified_seen = 0 # Records last time a potential user was detected
greeted = False # Keeps track of greetings so user is only greeted once
hard_reset = False # To reset the system when a user is detected
llm_process = None 

def start_llm():
    global llm_process
    with llm_lock:
        # Avoid starting duplicate llm's
        if llm_process is not None and llm_process.poll() is None:
            return
        
        llm_process = subprocess.Popen(
        [sys.executable, "llm_worker3.py"], # Ensure same python interpreter is running and start llm script
        #Set up communication pipeline
        stdin=subprocess.PIPE, # Send input to llm
        stdout=subprocess.PIPE, # Read response from llm
        stderr=subprocess.PIPE, #capture errors
        text=True, # send everything as text
        bufsize=1 # Line buffered (good for real time communication)
        )
    print("[LLM] Worker started") # Log

# Returns true if hard_reset and reset_in_progress is not active and expected_session matches session id
# Basically used track if a session is active
def _session_is_active(expected_session: int) -> bool:
    return (not hard_reset) and (not reset_in_progress) and session_id == expected_session

# cleanup function for speech
def _finish_speech(request_session: int):
    global talk

    with lock:
        pending_speeches[request_session] -= 1 # reduce count of pending speech by one
        if pending_speeches[request_session] <= 0: # if not speech left in session
            pending_speeches.pop(request_session, None) # Remove session from dictionary
        talk = pending_speeches.get(session_id, 0) > 0 # stays true until session has no speech left

# When user leaves reset session for next user
def hard_reset_system():
    global greeted, candidate_time, last_qualified_seen, is_listening, talk, hard_reset, llm_process, last_listen_time, listen_thread, session_id, reset_in_progress

    with lock:
        # Prevent overlapping resets i.e. if reset already started do nothing
        if reset_in_progress:
            return
        reset_in_progress = True
        hard_reset = True
        reset_session = session_id # Save session being reset
    
    sd.stop() # Stop the microphone

    # update LLM variables to shut it off
    with llm_lock:
        proc = llm_process
        llm_process = None

    # Terminate LLM worker
    if proc is not None and proc.poll() is None:
        proc.terminate()

    # Wait for any speech to finish first
    while True:
        with lock:
            if pending_speeches.get(reset_session, 0) == 0:
                break
        time.sleep(0.05)

    # Reset all session states
    with lock:
        session_id += 1
        greeted = False
        candidate_time = None
        last_qualified_seen = 0
        is_listening = False
        talk = False
        listen_thread = None
        last_listen_time = 0.0
        hard_reset = False
        reset_in_progress = False

    print ("[STATE] HARD RESET") # Log the hard reset in terminal

def submit_llm_process(user_text: str, request_session: int):
    global llm_process

    with lock:
        if not _session_is_active(request_session):
            return
        
    try:
        start_llm()
    except Exception as e:
        print("[LLM START ERROR]", e)
        speak("Sorry, I could not start the assistant.")
        return

    # Convert text to json
    payload = json.dumps({"text": user_text})

    llm_process.stdin.write(payload + "\n") # Write message to llm, \n signal end of message
    llm_process.stdin.flush() # Fiorce message to send immediately

    output = llm_process.stdout.readline().strip() # Wait for return message from llm and remove whitespaces

    # If no response is given from llm
    if not output:
        speak("Sorry, I could not process that request.")
        return
    
    with lock:
        if not _session_is_active(request_session):
            return
        
    # Convert response back from json
    data = json.loads(output)

    # Speak response
    if "response" in data:
        speak(data["response"])
    elif "error" in data:
        print("[LLM ERROR]", data["error"])
        speak("Sorry, I had trouble processing that request.")
    else:
        speak("Sorry, I had trouble processing that request.")

# Universal TTS (Works across platforms)
def _speak_universal(text: str, request_session: int):
    global talk, hard_reset

    # Make sure only one tts operation runs at a time
    with speech_lock:
        # Check speak request belongs to acive session
        with lock:
            should_skip = not _session_is_active(request_session)

        # If session is no longer valid, stop
        if should_skip:
            _finish_speech(request_session)
            return
        
        # Initialising tts engine and config for speech
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        # Speak the generated response
        engine.say(text)
        engine.runAndWait()

        _finish_speech(request_session)


# Start speech task in a background thread
def speak(text: str):
    global talk

    with lock:
        # Do not start speech during reset
        if hard_reset or reset_in_progress:
            return
        # Assign this speech to the current session
        request_session = session_id
        pending_speeches[request_session] += 1
        talk = True

    # Log speech in terminal
    print("[SPEAK]", text)
    # Start tts in background thread
    threading.Thread(target=_speak_universal, args=(text, request_session), daemon=True).start()

# Whisper listening and transcribing function
def _listen_and_transcribe(request_session: int):
    global is_listening, last_listen_time, listen_thread
    should_restart = False

    try:
        # Wait until system is not talking before listening
        while True:
            with lock:
                if not _session_is_active(request_session):
                    return
                if not talk:
                    break
            time.sleep(0.05)

        # Before recording, check ssession is valid
        with lock:
            if not _session_is_active(request_session):
                return
            
        # Log state in terminal and start recording from microphone
        print("[WHISPER] Listening...")
        audio = sd.rec(
            int(LISTEN_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        # Check session is still the same after recording, if not exit
        with lock:
            if not _session_is_active(request_session):
                return

        audio = np.squeeze(audio)

        # Log state in terminal and start transcribing with whisper
        print("[WHISPER] Transcribing...")
        result = whisper_model.transcribe(audio, fp16=False)
        text = result["text"].strip()

        # Check session is still the same after transcribing, if not exit
        with lock:
            if not _session_is_active(request_session):
                return
            
        if text:
            # Log user input in terminal and send to llm for processing
            print(f"[USER SAID] {text}")
            submit_llm_process(text, request_session)

            # wait until LLM speech finishes
            while True:
                with lock:
                    if not _session_is_active(request_session):
                        return
                    if not talk:
                        break
                time.sleep(0.05)
            # Goodbye prompt for user
            speak(
                "Thank you for using PERCI. If you are finished, please leave the station and I will reset for the next user. "
                "If you have any further questions, please do not hesitate to ask."
            )

        else:
            # Log no speech detected and prompt user to try again, then restart listening process
            print("[USER SAID] No speech detected.")
            speak("I'm sorry, I didn't quite catch that. Could you please repeat the sentence?")

        should_restart = True

    except Exception as e:
        # Log any Whisper errors
        print(f"[WHISPER ERROR] {e}")

    finally:
        with lock:
            # update listening state for this session
            if request_session == session_id:
                is_listening = False
                listen_thread = None
                last_listen_time = time.time()

        if should_restart:
            # Wait until system is not talking before listening again
            while True:
                with lock:
                    if not _session_is_active(request_session):
                        return
                    if not talk:
                        break
                time.sleep(0.05)
            # Restart listening cycle
            start_listening()

def start_listening():
    global is_listening, listen_thread

    with lock:
        # Check if system is currently resetting or has enetered a hard reset state
        if hard_reset or reset_in_progress: 
            return # Exit
        # Prevents duplicate listening threads
        if is_listening and listen_thread is not None and listen_thread.is_alive():
            return # Exit, listening alaready in progress
        # Make sure cooldown time has passed before listening again
        if time.time() - last_listen_time < LISTEN_COOLDOWN:
            return # Exit
        is_listening = True # System currently listening
        request_session = session_id # Store session id to listening request

    # Background Thread
    listen_thread = threading.Thread(target=_listen_and_transcribe, args=(request_session,), daemon=True)
    listen_thread.start()

    
def main():
    global greeted, candidate_time, last_qualified_seen, talk, hard_reset
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
            last_qualified_seen = now # Store time last potewntial user was seen (updates every frame)

            if candidate_time is None: # Store time potewntial user first appears
                candidate_time = now

            if not greeted and (now - candidate_time >= HOLD_TIME): # If potential user has been present for lonmg enough they are now a user, if system hasent greeted them already do so (Prompt LLM to greet user)
                greeted = True
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

    with llm_lock:
        proc = llm_process
    if proc is not None and proc.poll() is None:
        proc.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_llm()
    main()