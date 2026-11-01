# Image and Speech Processing Labs

Python programs for image and speech processing lab exercises.

## Install requirements

Run this command in the project folder (vs code terminal):

```powershell
py -m pip install opencv-python numpy matplotlib scikit-image scipy PyWavelets SpeechRecognition pyttsx3 sounddevice soundfile
```

For `Lab12.py` offline speech recognition, install PocketSphinx:

```powershell
py -m pip install --only-binary=:all: pocketsphinx
```

If PocketSphinx fails, the other packages above remain installed.
