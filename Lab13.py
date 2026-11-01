"""
Required packages, command: pip install pyttsx3 sounddevice soundfile numpy matplotlib
"""

import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# ---------------- TEXT TO SPEECH ----------------
text = "Welcome to Speech Processing Lab."

engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()

# ---------------- SPEECH RECORDING ----------------
fs = 44100          # Sampling frequency
duration = 5        # Recording time in seconds

print("\nRecording will start now...")
print("Speak clearly for 5 seconds.")

recording = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1,
    dtype='float32'
)

sd.wait()   # Wait until recording is finished

print("Recording finished.")

# Save the recorded speech
filename = "recorded_speech.wav"
sf.write(filename, recording, fs)

print("Speech saved as:", filename)

# ---------------- PLAY RECORDED SPEECH ----------------
print("Playing recorded speech...")
sd.play(recording, fs)
sd.wait()

# ---------------- PLOT SPEECH SIGNAL ----------------
time = np.arange(len(recording)) / fs

plt.figure(figsize=(10, 4))
plt.plot(time, recording)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Recorded Speech Signal")
plt.grid()
plt.show()