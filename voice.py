from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import time


class VoiceController:

    def __init__(self):
        print("Loading Whisper model...")

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        print("Whisper Loaded!")

    def listen(self):

        fs = 16000
        duration = 3

        print("\n🎤 Listening...")

        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        # Create temporary audio file
        temp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_name = temp.name
        temp.close()   # IMPORTANT for Windows

        # Save recording
        write(temp_name, fs, recording)

        # Convert speech to text
        segments, info = self.model.transcribe(temp_name)

        text = ""

        for segment in segments:
            text += segment.text

        text = text.lower().strip()

        print("You said:", text)

        # Give Whisper a moment to release the file
        time.sleep(0.5)

        # Delete temp file safely
        try:
            os.remove(temp_name)
        except PermissionError:
            pass

        return text