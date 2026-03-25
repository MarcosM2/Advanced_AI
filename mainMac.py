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

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Load Whisper
whisper_model = whisper.load_model("base")

# Open webcam, typically index 0
cap = cv2.VideoCapture(0)

# Set trigger rules deciding when someone is a real user
MIN_BOX_AREA = 120000  # Minimum bounding box area to consider a person a potential user
HOLD_TIME = 3.0        # Time a potential user must wait before being considered a user
RESET_TIME = 2.0       # Time after the user leaves before system resets

# Whisper audio settings
SAMPLE_RATE = 16000
LISTEN_SECONDS = 4
LISTEN_COOLDOWN = 6.0

# Shared state for listening
lock = threading.Lock()
is_listening = False
talk = False
last_listen_time = 0.0

candidate_time = None      # Records time when a potential user was first detected
last_qualified_seen = 0    # Records last time a potential user was detected
greeted = False            # Keeps track of greetings so user is only greeted once
hard_reset = False         # To reset the system when a user is detected
speech_process = None
llm_process = None


def start_llm():
    global llm_process
    # Avoid starting duplicate llm's
    if llm_process is not None and llm_process.poll() is None:
        return

    llm_process = subprocess.Popen(
        [sys.executable, "llm_worker3.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    print("[LLM] Worker started")


def hard_reset_system():
    global greeted, candidate_time, last_qualified_seen, is_listening, talk, hard_reset, speech_process, llm_process, last_listen_time

    # Wait for any current speech to finish
    while True:
        with lock:
            if not talk:
                break
        time.sleep(0.05)

    # Speak thank you message before resetting
    speak(
        "Thank you for using PERCI. If you are finished, please leave the station and I will reset for the next user. "
        "If you have any further questions, please do not hesitate to ask."
    )

    # Small delay to allow the TTS thread to start and set talk = True
    time.sleep(0.2)

    # Wait for the thank you message to finish
    while True:
        with lock:
            if not talk:
                break
        time.sleep(0.05)

    with lock:
        hard_reset = True

    sd.stop()  # Stop the microphone

    if llm_process is not None:
        llm_process.terminate()

    with lock:
        greeted = False
        candidate_time = None
        last_qualified_seen = 0
        is_listening = False
        talk = False
        speech_process = None
        llm_process = None
        last_listen_time = 0

    print("[STATE] HARD RESET")


def submit_llm_process(user_text: str):
    global llm_process

    if llm_process is None or llm_process.poll() is not None:
        start_llm()

    payload = json.dumps({"text": user_text})
    llm_process.stdin.write(payload + "\n")
    llm_process.stdin.flush()

    output = llm_process.stdout.readline().strip()

    with lock:
        if hard_reset:
            return

    if not output:
        speak("Sorry, I could not process that request.")
        return

    data = json.loads(output)

    if "response" in data:
        speak(data["response"])
    elif "error" in data:
        print("[LLM ERROR]", data["error"])
        speak("Sorry, I had trouble processing that request.")
    else:
        speak("Sorry, I had trouble processing that request.")


def _speak_universal(text: str):
    global talk, hard_reset

    with lock:
        if hard_reset:
            return
        talk = True

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print("[TTS ERROR]", e)

    finally:
        with lock:
            talk = False


def speak(text: str):
    with lock:
        if hard_reset:
            return

    print("[SPEAK]", text)
    threading.Thread(target=_speak_universal, args=(text,), daemon=True).start()


def _listen_and_transcribe():
    global is_listening, last_listen_time
    should_restart = False

    try:
        # Wait until PERCI is not talking before listening
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
            submit_llm_process(text)

            # Small delay to allow the TTS thread to start and set talk = True
            time.sleep(0.2)

            # Wait until LLM speech finishes
            while True:
                with lock:
                    if hard_reset:
                        return
                    if not talk:
                        break
                time.sleep(0.05)

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
    global greeted, candidate_time, last_qualified_seen, talk, hard_reset

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not working")
            break

        results = model(frame, classes=[0], verbose=False)
        result = results[0]

        largest_box = None
        largest_area = 0

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        now = time.time()

        if largest_box is not None and largest_area >= MIN_BOX_AREA:
            with lock:
                if hard_reset:
                    hard_reset = False
                    print("[STATE] READY FOR NEXT USER")

            last_qualified_seen = now

            if candidate_time is None:
                candidate_time = now

            if not greeted and (now - candidate_time >= HOLD_TIME):
                greeted = True
                candidate_time = None
                speak("Hello I am PERCI, an AI assistant designed to help you find your way around this building. Please tell me, where would you like to go?")
                time.sleep(10)
                start_listening()
        else:
            candidate_time = None

            if greeted and (now - last_qualified_seen >= RESET_TIME):
                hard_reset_system()

        frame = result.plot()

        if largest_box is not None:
            x1, y1, x2, y2 = largest_box
            cv2.putText(
                frame,
                f"Largest area: {largest_area}",
                (x1 + 150, max(30, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        if greeted and largest_box is not None:
            x1, y1, x2, y2 = largest_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Fixed: removed stray comma
            cv2.putText(
                frame,
                "User",
                (x1, max(30, y1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        cv2.imshow("User Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # Clean up
    sd.stop()

    if speech_process is not None:
        speech_process.terminate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_llm()
    main()
