from voice import VoiceController

voice = VoiceController()

while True:
    text = voice.listen()
    print(text)