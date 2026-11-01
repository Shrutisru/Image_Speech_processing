"""
Q10: Separation of voiced / unvoiced / silence regions of a speech signal
Method: Short-Time Energy (STE) + Zero Crossing Rate (ZCR)

    Silence  -> very low energy
    Voiced   -> high energy, LOW  zero-crossing rate
    Unvoiced -> lower energy, HIGH zero-crossing rate (noise-like)

Requires: numpy, scipy, matplotlib   (
command:  pip install numpy scipy matplotlib  
or use this command: py -m pip install numpy matplotlib scipy
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ---------- 1. Read the signal ----------
fs, x = wavfile.read('audio.wav')
if x.ndim > 1:                      # stereo -> mono
    x = x.mean(axis=1)
x = x.astype(float)
x = x / np.max(np.abs(x))           # normalise to [-1, 1]
t = np.arange(len(x)) / fs

# ---------- 2. Framing (25 ms frame, 10 ms hop) ----------
frame_len = int(0.025 * fs)
hop = int(0.010 * fs)
n_frames = 1 + (len(x) - frame_len) // hop

energy = np.zeros(n_frames)
zcr = np.zeros(n_frames)
for i in range(n_frames):
    seg = x[i * hop: i * hop + frame_len]
    energy[i] = np.sum(seg ** 2)                          # short-time energy
    zcr[i] = 0.5 * np.mean(np.abs(np.diff(np.sign(seg))))  # zero-crossing rate

# ---------- 3. Thresholds (tune these for your recording) ----------
E_thr = 0.02 * np.max(energy)       # below this  -> silence
Z_thr = 0.15                        # above this  -> unvoiced, else voiced

# ---------- 4. Classify each frame: 0 = silence, 1 = unvoiced, 2 = voiced ----------
label = np.zeros(n_frames, dtype=int)
for i in range(n_frames):
    if energy[i] < E_thr:
        label[i] = 0
    elif zcr[i] > Z_thr:
        label[i] = 1
    else:
        label[i] = 2

# ---------- 5. Expand frame labels to sample level ----------
sample_label = np.zeros(len(x), dtype=int)
for i in range(n_frames):
    sample_label[i * hop: i * hop + hop] = label[i]
sample_label[n_frames * hop:] = label[-1]

# ---------- 6. Plots ----------
names = ['Silence', 'Unvoiced', 'Voiced']
colors = ['gray', 'red', 'blue']

plt.figure(figsize=(11, 8))

plt.subplot(4, 1, 1)
plt.plot(t, x, 'k')
plt.title('Original speech signal'); plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(energy); plt.axhline(E_thr, color='r', ls='--', label='Energy threshold')
plt.title('Short-time energy'); plt.legend()

plt.subplot(4, 1, 3)
plt.plot(zcr); plt.axhline(Z_thr, color='r', ls='--', label='ZCR threshold')
plt.title('Zero-crossing rate'); plt.legend()

plt.subplot(4, 1, 4)
for c in range(3):
    y = np.where(sample_label == c, x, np.nan)
    plt.plot(t, y, color=colors[c], label=names[c])
plt.title('Voiced / Unvoiced / Silence regions')
plt.xlabel('Time (s)'); plt.legend(loc='upper right')

plt.tight_layout()
plt.show()