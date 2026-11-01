"""
Q12 (OFFLINE): Recognize a speech signal using PocketSphinx

One-time setup (needs internet only once):
    pip install SpeechRecognition
    python -m pip install --only-binary=:all: pocketsphinx
    
After that it works fully offline. English only.
audio.wav must be a normal PCM wav file.
"""
import speech_recognition as sr

recognizer = sr.Recognizer()

# 1. Load the audio file
with sr.AudioFile('audio.wav') as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.2) 
    audio = recognizer.record(source)                          

# 2. Recognize offline with PocketSphinx
try:
    text = recognizer.recognize_sphinx(audio)
    print('Recognized text:')
    print(text)
except sr.UnknownValueError:
    print('Speech was not understood (try a clearer recording).')
except sr.RequestError as e:
    print('PocketSphinx error (is pocketsphinx installed?):', e)