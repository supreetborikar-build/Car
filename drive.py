
import base64
from io import BytesIO
import threading
import time

import cv2
import numpy as np
from PIL import Image

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import keras
from tensorflow.keras.models import load_model
from voice import VoiceController

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("model.h5")

# -----------------------------
# Voice Controller
# -----------------------------
voice = VoiceController()

# -----------------------------
# Socket Server
# -----------------------------
sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 20
MIN_SPEED = 10

# -----------------------------
# Manual Control Variables
# -----------------------------
manual_mode = False
manual_steering = 0.0
manual_throttle = 0.0

voice_command = ""


# -----------------------------
# Voice Listener Thread
# -----------------------------
def voice_listener():

    global manual_mode
    global manual_steering
    global manual_throttle
    global voice_command

    while True:

        try:
            command = voice.listen()

            if not command:
                continue

            voice_command = command

            print(f"\n🎤 Command: {command}")

            if "manual" in command:
                manual_mode = True
                print("🟢 Manual Mode Enabled")

            elif "auto" in command:
                manual_mode = False
                print("🤖 AI Mode Enabled")

            elif "forward" in command or "move" in command:

                manual_throttle = 0.35

            elif "stop" in command:

                manual_throttle = 0.0

            elif "left" in command:

                manual_steering = -0.6

            elif "right" in command:

                manual_steering = 0.6

            elif "straight" in command:

                manual_steering = 0.0

        except Exception as e:

            print("Voice Error:", e)

        time.sleep(0.1)


# Start Voice Thread
threading.Thread(
    target=voice_listener,
    daemon=True
).start()

# -----------------------------
# Simulator Connected
# -----------------------------
@sio.event
def connect(sid, environ):
    print("🚗 Simulator Connected")
    send_control(0, 0)


# -----------------------------
# Telemetry
# -----------------------------
@sio.on("telemetry")
def telemetry(sid, data):

    global manual_mode
    global manual_steering
    global manual_throttle

    if data is None:
        return

    # Current Speed
    speed = float(data["speed"])

    # Camera Image
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)

    # Same preprocessing used during training
    image = image[60:135, :, :]
    image = cv2.resize(image, (200, 66))
    image = np.expand_dims(image, axis=0)

    # -----------------------------
    # AI Prediction
    # -----------------------------
    ai_steering = float(model.predict(image, verbose=0))

    if speed > MAX_SPEED:
        ai_throttle = 0.10
    elif speed < MIN_SPEED:
        ai_throttle = 0.50
    else:
        ai_throttle = 0.30

    # -----------------------------
    # Manual / AI Mode
    # -----------------------------
    if manual_mode:

        steering = manual_steering
        throttle = manual_throttle
        mode = "🎤 MANUAL"

    else:

        steering = ai_steering
        throttle = ai_throttle
        mode = "🤖 AI"

    print(
        f"{mode} | "
        f"Speed: {speed:.2f} | "
        f"Steering: {steering:.3f} | "
        f"Throttle: {throttle:.2f}"
    )

    send_control(steering, throttle)


# -----------------------------
# Send Controls
# -----------------------------
def send_control(steering, throttle):

    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle)
        }
    )


# -----------------------------
# Start Server
# -----------------------------
if __name__ == "__main__":

    app = socketio.WSGIApp(sio, app)

    print("=" * 60)
    print("🚗 Self Driving Car Started")
    print("🎤 Whisper Voice Control Enabled")
    print("🤖 AI Driving Ready")
    print("Waiting for simulator on port 4567...")
    print("=" * 60)

    eventlet.wsgi.server(
        eventlet.listen(("", 4567)),
        app
    )