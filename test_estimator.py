import numpy as np 
from CST_blocks_CSPB import SSCA_Compute, Compute_Coherence_function, estimate_psd_fsm
import matplotlib.pyplot as plt

# Parameters
T_bit = 10               # Bit duration
num_bits = 6554          # Number of bits
f_offset = 0.05               # Normalized carrier frequency
N0_dB = -10.0           # Noise power spectral density in dB
Power_dB = 0.0          # Signal power in dB
N_psd = 65536        # Number of FFT points in PSD

# Generate bit sequence
symbols = np.random.randint(0, 2, num_bits) * 2 - 1 # random 1's and -1's
bpsk = np.repeat(symbols, T_bit)  # repeat each symbol sps times to make rectangular BPSK
# Frequency shift by multiplying with complex exponential
bpsk = bpsk * np.exp(2j * np.pi * f_offset * np.arange(len(bpsk)))
print(np.shape(bpsk))

# Plot time-domain waveform
# plt.figure(figsize=(8, 3))
# plt.plot(bpsk[:200], linewidth=2)
# plt.ylim([-1.2, 1.2])
# plt.grid(True)
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Amplitude')
# plt.title(r'Time-Domain Plot of Rectangular-Pulse BPSK ($T_{bit} = 10, f_c = 0$)')
# plt.savefig('rect_bpsk_time_domain.jpg')
#plt.show()



# Generate noise
noise = np.random.randn(len(bpsk)) + 1j * np.random.randn(len(bpsk))
print(np.shape(noise))
N0_linear = 10 ** (N0_dB / 10)   # Convert N0_dB to linear scale
noise_power = np.var(noise)      # Measure noise variance
print(noise_power)
pow_factor = np.sqrt(N0_linear / noise_power)  # Scale noise power
#pow_factor=0.1
print("noise_power, pow_factor", noise_power, pow_factor)
noise *= pow_factor

# Add noise to signal
data_samples = bpsk + noise
print(np.sort(np.abs(data_samples))[-20:][::-1])
# Compute PSD
num_blocks = len(data_samples) // N_psd
S = data_samples[:num_blocks * N_psd].reshape(N_psd, num_blocks, order='F')
#print(S.shape)
I = np.fft.fft(S, axis=0)
#print(I[:10])
I = np.abs(I) ** 2
I = np.sum(I, axis=1)  # Average over blocks
print(I.shape)
I = np.fft.fftshift(I)  # Shift DC to center
#print(I)
I = I / (num_blocks * N_psd)  # Normalize
#direct_fft=np.abs(np.fft.fftshift(np.fft.fft(data_samples)))
# Frequency vector
freq_vec = np.linspace(-0.5, 0.5,N_psd)

# Plot PSD
plt.figure(figsize=(8, 3))
plt.plot(freq_vec, 10 * np.log10(I), linewidth=2)
plt.grid(True)
plt.xlabel('Frequency (Normalized)')
plt.ylabel('PSD (dB)')
plt.title(r'Estimated Power Spectrum for inside SSCA')
plt.savefig('rect_bpsk_psd.jpg')
plt.show()

# Power Check
meas_pow = np.sum(I) * (1.0 / N_psd)
Power_linear = 10 ** (Power_dB / 10.0)
print(f"PSD-measured power: {meas_pow:.5e}, Known total power: {(N0_linear + Power_linear):.5e}")
SCF_conj, SCF_nonconj =SSCA_Compute(data_samples[0:65536])

cyclic_frequencies = np.linspace(-1, 1, SCF_conj.shape[0])  # Approximate cyclic frequencies
print("zero_cyclic frequency location", np.argmin(np.abs(cyclic_frequencies)))
baseband_frequencies = np.linspace(-0.5, 0.5, SCF_conj.shape[1])
center_idx = len(baseband_frequencies) // 2  
plt.figure(figsize=(10, 6))
#plt.imshow(10 * np.log10(SCF + 1e-10), aspect='auto',  
#       extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], origin='lower', vmin=np.percentile(10 * np.log10(SCF + 1e-10), 5), vmax=np.percentile(10 * np.log10(SCF + 1e-10), 99))
image=plt.imshow(np.abs(SCF_conj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_conj))/100, origin='lower')
#image=plt.imshow(np.abs(SCF_nonconj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_conj))/100, origin='lower')
#image=plt.imshow(np.abs(SCF_non_conj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF))/100, origin='lower')
plt.colorbar(image, label="SCF Magnitude (dB)")
plt.xlabel("Baseband Frequency (Hz)")
plt.ylabel("Cyclic Frequency (Hz)")
plt.title("Conjugate Spectral Correlation Function (SCF)")
plt.show()

plt.figure(figsize=(8, 3))
image=plt.imshow(np.abs(SCF_conj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_conj))/100, origin='lower')
#image=plt.imshow(np.abs(SCF_nonconj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_conj))/100, origin='lower')
image=plt.imshow(np.abs(SCF_nonconj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_nonconj))/100, origin='lower')
plt.colorbar(image, label="SCF Magnitude (dB)")
plt.xlabel("Baseband Frequency (Hz)")
plt.ylabel("Cyclic Frequency (Hz)")
plt.title("Non-conjugate spectral Correlation Function (SCF)")
plt.show()

flat_indices_nonconj = np.argsort(SCF_nonconj.ravel())[-500:]  # Get indices of top 500 largest values
#print("flat_indices_nonconj", flat_indices_nonconj)

# Step 2: Convert the flattened indices back to row (i) and column (j) indices_rp
i_indices, j_indices = np.unravel_index(flat_indices_nonconj, SCF_nonconj.shape)
#print(i_indices.shape, j_indices.shape)
#print(i_indices.max(), j_indices.max())
# Step 3: Normalize the indices to fit the plot limits
# For the x-axis, scale the column indices to range [-0.5, 0.5]
# Convert indices to actual frequency values
alpha_values = -1 + (i_indices / (SCF_nonconj.shape[0]- 1)) * 2  # Map row indices to alpha range (-1 to 1)
f_values = -0.5 + (j_indices / (SCF_nonconj.shape[1] - 1)) * 1        # Map column indices to f range (-0.5 to 0.5)
print(alpha_values.max(), f_values.max(), alpha_values.min(), alpha_values.min())
#print(i_indices, j_indices)
#print(SCF_nonconj[i_indices,j_indices])
# Scatter plot in f-alpha plane
plt.figure(figsize=(8, 6))
plt.scatter(f_values, alpha_values, color='b', marker='.', alpha=0.8)  # Blue dots

# Labels and title
plt.xlabel("Frequency (Normalized Hz)")
plt.ylabel("Cycle Frequency (Normalized Hz)")
plt.title("Non Conjugate Cycle Frequency 500 Largest SCF Mags")

# Formatting grid
plt.grid(True, linestyle='--', alpha=0.6)

# Ensure axis limits match expected ranges
plt.xlim([-0.5, 0.5])
plt.ylim([0, 1])
plt.xticks(np.arange(-0.5, 0.6, 0.1))
plt.yticks(np.arange(0, 1, 0.1))
# Show the plot
plt.show()

flat_indices_conj = np.argsort(SCF_conj.ravel())[-500:]  # Get indices of top 500 largest values

# Step 2: Convert the flattened indices back to row (i) and column (j) indices
i_indices, j_indices = np.unravel_index(flat_indices_conj, SCF_conj.shape)
#print(i_indices.shape, j_indices.shape)
#print(i_indices.max(), j_indices.max())
# Step 3: Normalize the indices to fit the plot limits
# For the x-axis, scale the column indices to range [-0.5, 0.5]
# Convert indices to actual frequency values
alpha_values = -1 + (i_indices / (SCF_conj.shape[0]- 1)) * 2  # Map row indices to alpha range (-1 to 1)
f_values = -0.5 + (j_indices / (SCF_conj.shape[1] - 1)) * 1        # Map column indices to f range (-0.5 to 0.5)
#print(alpha_values.max(), f_values.max(), alpha_values.min(), alpha_values.min())
#print(alpha_values, f_values)
# Scatter plot in f-alpha plane
plt.figure(figsize=(8, 6))
plt.scatter(f_values, alpha_values, color='b', marker='.', alpha=0.8)  # Blue dots

# Labels and title
plt.xlabel("Frequency (Normalized Hz)")
plt.ylabel("Cycle Frequency (Normalized Hz)")
plt.title("Conjugate Cycle Frequency 500 Largest SCF Mags")

# Formatting grid
plt.grid(True, linestyle='--', alpha=0.6)

# Ensure axis limits match expected ranges
plt.xlim([-0.5, 0.5])
plt.ylim([-1, 1])
plt.xticks(np.arange(-0.5, 0.6, 0.1))
plt.yticks(np.arange(-1, 1, 0.2))
# Show the plot
plt.show()

#### Compute coherence and plot it. 
#print(len(f_values), len(alpha_values))

# Plot PSD
f,PSD_estimate = estimate_psd_fsm(data_samples, N_psd)
plt.figure(figsize=(8, 3))
plt.plot(freq_vec, 10 * np.log10(I), linewidth=2)
plt.plot(f,10*np.log10(PSD_estimate), linewidth=2)
#lt.plot(baseband_frequencies, 10*np.log10(SCF_nonconj[32767,:]), linewidth=2)
plt.grid(True)
plt.xlabel('Frequency (Normalized)')
plt.ylabel('PSD (dB)')
plt.title(r'Estimated Power Spectrum for Rectangular-Pulse BPSK ($T_{bit} = 10, f_c = 0.05$)')
plt.savefig('rect_bpsk_psd.jpg')
plt.show()

############################ Non-conjugate coherence function

rho, rho_conj = Compute_Coherence_function(SCF_nonconj, SCF_conj, PSD_estimate, baseband_frequencies, cyclic_frequencies, N_psd)
max_mag_arg=np.unravel_index(np.argmax(rho), rho.shape)
#print("rho_max_mag and SCF_max_mag", max_mag_arg, SCF_nonconj[max_mag_arg], np.max(SCF_nonconj) )
print("Rho_max and Rho_min, PSD_max and PSD_min", rho.max(), rho.min(), I.max(), I.min(), rho_conj.max(), rho_conj.min())
#print("SCF(PSF)", SCF_nonconj[32668,:]-SCF_nonconj[32667,:], SCF_nonconj[32668,:]-SCF_nonconj[32669,:], SCF_nonconj[32668,:]-SCF_nonconj[32670,:])
flat_indices_nonconj = np.argsort(rho.ravel())[-500:]  # Get indices of top 500 largest values

# Step 2: Convert the flattened indices back to row (i) and column (j) indices

i_indices, j_indices = np.unravel_index(flat_indices_nonconj, SCF_conj.shape)
#print(i_indices, j_indices)
#print(i_indices.max(), j_indices.max())
# Step 3: Normalize the indices to fit the plot limits
# For the x-axis, scale the column indices to range [-0.5, 0.5]
# Convert indices to actual frequency values
alpha_values_nonconj = -1 + (i_indices / (rho.shape[0]- 1)) * 2  # Map row indices to alpha range (-1 to 1)
f_values_nonconj = -0.5 + (j_indices / (rho.shape[1] - 1)) * 1        # Map column indices to f range (-0.5 to 0.5)

plt.figure(figsize=(8, 6))
plt.scatter(f_values_nonconj, alpha_values_nonconj, color='b', marker='.', alpha=0.8)  # Blue dots

# Labels and title
plt.xlabel("Frequency (Normalized Hz)")
plt.ylabel("Cycle Frequency (Normalized Hz)")
plt.title("Non Conjugate Cycle Frequency vs Freq for 500 Largest Coherence Mags")

# Formatting grid
plt.grid(True, linestyle='--', alpha=0.6)

# Ensure axis limits match expected ranges
plt.xlim([-0.5, 0.5])
plt.ylim([0, 1])
plt.xticks(np.arange(-0.5, 0.6, 0.1))
plt.yticks(np.arange(0, 1, 0.1))
# Show the plot
plt.show()

##### Conjugate Coherence Function
#rho_conj = Compute_Coherence_function(SCF_conj, PSD_estimate, baseband_frequencies, cyclic_frequencies, N_psd)
flat_indices_conj = np.argsort(rho_conj.ravel())[-500:]  # Get indices of top 500 largest values

# Step 2: Convert the flattened indices back to row (i) and column (j) indices
i_indices_conj, j_indices_conj = np.unravel_index(flat_indices_conj, SCF_conj.shape)
#print(i_indices, j_indices)
#print(i_indices.max(), j_indices.max())
# Step 3: Normalize the indices to fit the plot limits
# For the x-axis, scale the column indices to range [-0.5, 0.5]
# Convert indices to actual frequency values
alpha_values_conj = -1 + (i_indices_conj / (rho_conj.shape[0]- 1)) * 2  # Map row indices to alpha range (-1 to 1)
f_values_conj = -0.5 + (j_indices_conj / (rho_conj.shape[1] - 1)) * 1        # Map column indices to f range (-0.5 to 0.5)
#print(alpha_values.max(), f_values.max(), alpha_values.min(), alpha_values.min())
# Scatter plot in f-alpha plane
plt.figure(figsize=(8, 6))
plt.scatter(f_values_conj, alpha_values_conj, color='b', marker='.', alpha=0.8)  # Blue dots

# Labels and title
plt.xlabel("Frequency (Normalized Hz)")
plt.ylabel("Cycle Frequency (Normalized Hz)")
plt.title("Conjugate Cycle Frequency vs Freq for 500 Largest Coherence Mags")

# Formatting grid
plt.grid(True, linestyle='--', alpha=0.6)

# Ensure axis limits match expected ranges
plt.xlim([-0.5, 0.5])
plt.ylim([-1, 1])
plt.xticks(np.arange(-0.5, 0.6, 0.1))
plt.yticks(np.arange(-1, 1, 0.2))
# Show the plot
plt.show()

