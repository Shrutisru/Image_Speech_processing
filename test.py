"""
Q10: Separation of Voiced, Unvoiced and Silence
Method: Short-Time Energy (STE) + Zero Crossing Rate (ZCR)

Install: py -m pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Read audio
fs, x = wavfile.read("audio.wav")

if x.ndim > 1:
    x = x.mean(axis=1)

x = x.astype(float)
x = x / np.max(np.abs(x))
t = np.arange(len(x)) / fs

# 2. Framing
frame_len = int(0.025 * fs)
hop = int(0.010 * fs)
n_frames = 1 + (len(x) - frame_len) // hop

# 3. Calculate STE and ZCR
energy = np.zeros(n_frames)
zcr = np.zeros(n_frames)

for i in range(n_frames):
    seg = x[i * hop : i * hop + frame_len]

    energy[i] = np.sum(seg ** 2)
    zcr[i] = 0.5 * np.mean(np.abs(np.diff(np.sign(seg))))

# 4. Thresholds
E_thr = 0.02 * np.max(energy)
Z_thr = 0.15

# 5. Classification
label = np.zeros(n_frames, dtype=int)

for i in range(n_frames):
    if energy[i] < E_thr:
        label[i] = 0
    elif zcr[i] > Z_thr:
        label[i] = 1
    else:
        label[i] = 2

# 6. Convert frame labels to sample labels
sample_label = np.repeat(label, hop)
sample_label = np.pad(sample_label, (0, len(x) - len(sample_label)),
                      constant_values=label[-1])

# 7. Plot results
names = ['Silence', 'Unvoiced', 'Voiced']
colors = ['gray', 'red', 'blue']

plt.figure(figsize=(11, 8))

plt.subplot(4, 1, 1)
plt.plot(t, x, 'k')
plt.title('Original speech signal')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(energy)
plt.axhline(E_thr, color='r', ls='--', label='Energy threshold')
plt.title('Short-time energy')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(zcr)
plt.axhline(Z_thr, color='r', ls='--', label='ZCR threshold')
plt.title('Zero-crossing rate')
plt.legend()

plt.subplot(4, 1, 4)
for c in range(3):
    y = np.where(sample_label == c, x, np.nan)
    plt.plot(t, y, color=colors[c], label=names[c])

plt.title('Voiced / Unvoiced / Silence regions')
plt.xlabel('Time (s)')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()