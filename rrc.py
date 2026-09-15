import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt

#1. Random Bit Generator
rng = np.random.default_rng(122807528840384100672342137672332424406)
bitstream = rng.integers(0,1,10000,endpoint=True)

#BPSK Mapping
bpsk = np.where(bitstream == 0,1,-1)

#2. Upsampling (L=4)
ovf = 4
ov_signal = np.zeros(len(bpsk)*ovf)
ov_signal[::ovf] = bpsk

#3. Pulse shaping
Tsym = 1
Nsym = 8
B = 0.5

num_points = (Nsym * ovf) + 1
t = np.linspace(-Nsym / 2, Nsym / 2, num_points)
p = np.zeros_like(t)

#SSRC Impulse Response
for i,ti in enumerate(t):
    if np.isclose(ti, 0.0):
        p[i] = (1 / np.sqrt(Tsym)) * (1 + B * (4 / np.pi - 1))
        
    elif np.isclose(np.abs(ti), Tsym / (4 * B)):
        term1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * B))
        term2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * B))
        p[i] = (B / (np.sqrt(2 * Tsym))) * (term1 + term2)
    else:
        num_term1 = np.cos((1 + B) * np.pi * ti / Tsym)
        num_term2 = (Tsym / (4 * B * ti)) * np.sin((1 - B) * np.pi * ti / Tsym)
        den = 1 - (4 * B * ti / Tsym) ** 2
        
        p[i] = (1 / np.sqrt(Tsym)) * (4 * B / np.pi) * (num_term1 + num_term2) / den

shaped_sig = signal.convolve(ov_signal,p,mode='same')

#4. AWGN
snr_range = np.arange(0, 17, 2)
received_s = []
for snr_db in snr_range:
    snr_lin = 10**(snr_db/10)
    power_s = np.mean(shaped_sig ** 2)
    power_n = power_s / snr_lin
    sigma2 = power_n
    n = np.random.normal(0, np.sqrt(sigma2), len(shaped_sig))
    received_s.append(shaped_sig+n)

#5. Matched Filtering
g = p[::-1]
for i in range(len(received_s)):
    received_s[i] = signal.convolve(received_s[i], g, mode='same')

# #6. Delay Compensation
# TD = num_points - 1
# for i in range(len(received_s)):
#     received_s[i] = received_s[i][TD+1:]

#7. Downsample
downsampled_sig = []
for signals in received_s:
    nsig = signals[::ovf]
    downsampled_sig.append(nsig)

#8. BPSK Demapping
demapped_bits = []
for signals in downsampled_sig:
    demapped_bits.append(np.where(signals>0,0,1))

#9. Compute BER
ber_simulated = []
for est_bits in demapped_bits:
    min_len = min(len(bitstream), len(est_bits))
    bit_errors = np.sum(bitstream[:min_len] != est_bits[:min_len])
    ber_simulated.append(bit_errors / min_len)

#10. Plot BER
plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ber_simulated, 'o-', label='Simulated BER')
plt.title('BPSK BER Performance over AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()