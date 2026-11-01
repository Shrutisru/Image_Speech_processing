
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile

# 1. Read audio
fs, x = wavfile.read('audio.wav')

if x.ndim > 1:
    x = x.mean(axis=1)

x = x.astype(float)
x = x / np.max(np.abs(x))
t = np.arange(len(x)) / fs

# 2. DWT Decomposition
wavelet = 'db4'
levels = 4

coeffs = pywt.wavedec(x, wavelet, level=levels)

# 3. Reconstruct A4, D4, D3, D2, D1
signals = []

for i in range(len(coeffs)):
    c = [np.zeros_like(a) for a in coeffs]
    c[i] = coeffs[i]
    signals.append(pywt.waverec(c, wavelet)[:len(x)])

A4 = signals[0]
D4, D3, D2, D1 = signals[1:]

# 4. Find intermediate approximations
A1 = A4 + D4 + D3 + D2
A2 = A4 + D4 + D3
A3 = A4 + D4

approximations = [A1, A2, A3, A4]
details = [D1, D2, D3, D4]

# 5. Plot results
plt.figure(figsize=(11, 12))

plt.subplot(5, 2, 1)
plt.plot(t, x, 'k')
plt.title('Original signal')

for k in range(1, 5):

    plt.subplot(5, 2, 2*k + 1)
    plt.plot(t, approximations[k-1], 'b')
    plt.title(f'Approximation A{k} (0 - {fs/2**(k+1):.0f} Hz)')

    plt.subplot(5, 2, 2*k + 2)
    plt.plot(t, details[k-1], 'r')
    plt.title(f'Detail D{k} ({fs/2**(k+1):.0f} - {fs/2**k:.0f} Hz)')

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# 6. Check reconstruction
recon = A4 + D4 + D3 + D2 + D1

print('Max reconstruction error:', np.max(np.abs(x - recon)))

