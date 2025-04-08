import numpy as np 
import scipy.signal as signal
import os, time, re 
import subprocess, pdb #, h5py
from read_binary import read_binary
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.fft import fft, ifft
from scipy import signal

current_dir =os.getcwd()
parent_dir=os.path.dirname(current_dir)

def nextpow2(n):
    return int(np.ceil(np.log2(n)))

def plot_psd(data_samples, title=' ', N_psd=256):
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
    plt.plot(freq_vec, I, linewidth=2)
    plt.grid(True)
    plt.xlabel('Frequency (Normalized)')
    plt.ylabel('PSD (dB)')
    plt.title(title)
    plt.savefig('rect_bpsk_psd.jpg')
    plt.show()

def plot_fft(data_samples, title=' ', N_psd=256):
    #num_blocks = len(data_samples) // N_psd
    #S = data_samples[:num_blocks * N_psd].reshape(N_psd, num_blocks, order='F')
    #print(S.shape)
    I = np.fft.fft(data_samples)
    #print(I[:10])
    freq_vec = np.fft.fftfreq(len(data_samples))
    
    # Plot PSD
    plt.figure(figsize=(8, 3))
    plt.plot(freq_vec, np.abs(I), linewidth=2)
    plt.grid(True)
    plt.xlabel('Frequency (Normalized)')
    plt.ylabel('FFT_mag')
    plt.title(title)
    #plt.savefig('rect_bpsk_psd.jpg')
    plt.show()
        
def estimate_psd_fsm(signal, N_fft=128):
    """
    Estimate the Power Spectral Density (PSD) using Frequency Smoothing Method.

    Parameters:
        signal (numpy array): Input signal.
        N_fft (int): FFT length for periodogram estimation.
        smoothing_window (int): Number of frequency bins for smoothing.

    Returns:
        f (numpy array): Frequency axis.
        psd_smooth (numpy array): Smoothed PSD estimate.
    """
    # Compute the periodogram (raw PSD estimate)
    smoothing_window_length=N_fft/20 #5% of the N_fft
    X_f = np.fft.fftshift(np.fft.fft(signal, N_fft))  # Compute FFT and shift
    psd = np.abs(X_f) ** 2 / N_fft  # Compute power spectrum

    # Smooth the PSD using a moving average filter
    kernel = np.hamming(smoothing_window_length)
    kernel_normalized =kernel/np.sum(kernel)
    psd_smooth = np.convolve(psd, kernel_normalized, mode='same')  # Smoothing operation

    # Frequency axis
    f = np.linspace(-0.5, 0.5, N_fft, endpoint=False)  # Normalized frequency

    return f, psd_smooth


def Compute_Coherence_function(SCF_nonconj, SCF_conj, PSD, f_axis, alpha_axis, N):
    """
    Compute the SCF ratio in a fully vectorized manner.

    Parameters:
    - SCF: 2D numpy array of shape (num_alpha, num_f), the spectral correlation function matrix.
    - PSD: 1D numpy array of length N, the power spectral density computed with N-point FFT.
    - f_axis: 1D numpy array of length num_f, frequency values corresponding to SCF columns.
    - alpha_axis: 1D numpy array of length num_alpha, cyclic frequency values corresponding to SCF rows.
    - N: int, number of points used in FFT for PSD computation.

    Returns:
    - rho: 2D numpy array of shape (num_alpha, num_f), the normalized SCF ratio.
    """

    num_alpha, num_f = SCF_conj.shape
    
    # Create an extended frequency axis for PSD (assuming it corresponds to FFT bin locations)
    PSD_f_axis = np.linspace(f_axis[0], f_axis[-1], N)  # Full-resolution frequency grid for PSD
    print(len(f_axis),len(PSD_f_axis), len(PSD), PSD_f_axis.shape, PSD.shape)
    # Interpolate PSD onto the SCF frequency grid
    PSD_interp = np.interp(f_axis, PSD_f_axis, PSD)  # Interpolated PSD

    
    # Compute f ± α/2 for all α and f at once
    f_plus = f_axis[None, :] + alpha_axis[:, None] / 2  # Shape: (num_alpha, num_f)
    f_minus = f_axis[None, :] - alpha_axis[:, None] / 2  # Shape: (num_alpha, num_f)
    f_alpha_conj = alpha_axis[:, None] / 2 - f_axis[None, :]  # Shape: (num_alpha, num_f)
    
    # Interpolate PSD for these frequency shifts
    PSD_f_plus = np.interp(f_plus, PSD_f_axis, PSD, left=0,right=0)
    PSD_f_minus = np.interp(f_minus, PSD_f_axis, PSD, left=0, right=0)
    PSD_f_alpha_conj = np.interp(f_alpha_conj, PSD_f_axis, PSD, left=0, right=0)
    
    # Compute the denominator element-wise
    denominator_rho = np.sqrt(PSD_f_plus * PSD_f_minus)
    denominator_rho_conj = np.sqrt(PSD_f_plus * PSD_f_alpha_conj)
    print("Coherence_denominator_shape", denominator_rho.shape)
    print("Coherence_denominator_shape", denominator_rho_conj.shape)
    # Avoid division by zero
    denominator_rho[denominator_rho == 0] = np.nan
    denominator_rho_conj[denominator_rho_conj == 0] = np.nan
    # Compute the ratio (fully vectorized)
    rho = np.nan_to_num(SCF_nonconj/ denominator_rho, nan=0.0)
    rho_conj = np.nan_to_num(SCF_conj/denominator_rho_conj, nan=0.0)
    #pdb.set_trace()
    return rho, rho_conj    


def FAM_compute(data_sample):
   Np = 512 # Number of input channels, should be power of 2, This is actually the resolution of frequency domain resolution, as delta_f =1/T 
            # Actually this is not number of channels, but the number of input samples over which FFT is taken. 
   L = Np//4 # Offset between points in the same column at consecutive rows in the same channelization matrix. It should be chosen to be less than or equal to Np/4
   num_windows = (len(data_sample) - Np) // L + 1 # Num windows determines the cycle-frequency resolution, over which complex demodulates are estimated
   Pe = int(np.floor(int(np.log(num_windows)/np.log(2))))
   P = 2**Pe
   N = L*P

   # channelization
   xs = np.zeros((num_windows, Np), dtype=complex)
   for i in range(num_windows):
     xs[i,:] = data_sample[i*L:i*L+Np]
   xs2 = xs[0:P,:]

    # windowing
   xw = xs2 * np.tile(np.hanning(Np), (P,1))

    # first FFT
   XF1 = np.fft.fftshift(np.fft.fft(xw))

   # freq shift down
   f = np.arange(Np)/float(Np) - 0.5
   f = np.tile(f, (P, 1))
   t = np.arange(P)*L
   t = t.reshape(-1,1) # make it a column vector
   t = np.tile(t, (1, Np))
   XD = XF1 * np.exp(-2j*np.pi*f*t)

   # main calcs
   SCF = np.zeros((2*N, Np))
   Mp = N//Np//2
   for k in range(Np):
       for l in range(Np):
           XF2 = np.fft.fftshift(np.fft.fft(XD[:,k]*np.conj(XD[:,l]))) # second FFT
           i = (k + l) // 2
           a = int(((k - l) / Np + 1) * N)
           SCF[a-Mp:a+Mp, i] = np.abs(XF2[(P//2-Mp):(P//2+Mp)])**2
    
   return SCF 

def SSCA_Compute(data_sample, fs=1,DF=0.02, d_alpha=0.0001):
    
    N=len(data_sample)
    #Np = 2 ** nextpow2(fs / DF) # %Number of input channels, defined by the desired frequency resolution (DF) as Np=fs/DF, It must be a
                                #power of 2 to avoid truncation or zero-padding in the FFT routines;
    Np=int(2**((np.log2(N))-9))
    print("N and Np Values:", N, Np)                         
    
    #channelization
    #Zero-padding X on both sides with Np/2 zeros
    
    pad_length = Np // 2
    X_padded = np.pad(data_sample,(pad_length, pad_length), mode='constant')
    plot_psd(X_padded)
    # Create data_sample_copy and pad the last elements to zero
    # Initialize X and fill it with segments of data_sample_copy
    data_matrix = np.column_stack([X_padded[i:N + i] for i in range(Np)])   
    a=np.kaiser(Np,12)
    a_normalized = a / np.sqrt(np.sum(a**2))
    data_matrix_windowed = data_matrix*a_normalized #signal.windows.chebwin(Np, 180)# *#np.kaiser(Np,5)signal.windows.chebwin(Np, 100)
    print("data_matrix_shape, window_shape, data_matrix_windowed_shape", data_matrix.shape, a.shape, data_matrix_windowed.shape)
    print("Windowing Done")
    # Step 1: First FFT
    XF1 = np.fft.fftshift(np.fft.fft(data_matrix_windowed, axis=1), axes=1)
    print("FFT Done")
    # Step 2: Downconversion
    k = np.arange(-Np // 2, Np // 2)  # Range of k values
    n = np.arange(1, N + 1)  # Range of n values

    # Geneterate the Downconversion matrix E using broadcasting
    E = np.exp(-1j * 2 * np.pi * np.outer(n - 1, k) / Np)

    print("Downconvesion matrix shape", E.shape)
    # Step 3: Multiply XF1 with E
    XD = XF1 * E  # Element-wise multiplication
    XD_abs=np.abs(XD)
    top_20 = np.sort(XD_abs.flatten())[-50:][::-1]
    #print(top_20)
    #print("Downconversion Done", XD.shape)
    # Conjugate Vector creating
    start_ind =Np//2
    X_nonconj_section = np.conj(X_padded[start_ind:start_ind+N])
    X_conj_section = X_padded[start_ind:start_ind+N]
    #print("Conjugation done", X_conj_section.shape)
    X_conj = np.tile(X_conj_section[:, np.newaxis], (1, Np))
    print("Conjugation tiling done", X_conj.shape)
    X_nonconj = np.tile (X_nonconj_section[:, np.newaxis], (1, Np))
    #print("Preprocessing before 2nd FFT Done")
    ## Second FFT ## , norm='ortho', , norm='ortho'norm='ortho'
    second_window=np.kaiser(N,6) # second_window.reshape(N,1)
    second_window_normalized=second_window/np.sum(second_window) #second_window.reshape(N,1)
    XFFT2_conj = np.fft.fftshift(np.fft.fft(XD*X_conj*second_window_normalized.reshape(N,1), axis=0), axes=0)  # Shift zero frequency component to the center#1/N #signal.windows.chebwin(N, 90).reshape(N,1)
    XFFT2_nonconj = np.fft.fftshift(np.fft.fft(XD*X_nonconj*second_window_normalized.reshape(N,1), axis=0), axes=0) # Shift zero frequency component to the center#signal.windows.chebwin(N, 10).reshape(N,1)#*signal.windows.chebwin(N, at=100).reshape(N,1)
    
    M_conj = np.abs(XFFT2_conj)
    M_nonconj = np.abs(XFFT2_nonconj)
    print(np.max(M_conj))
    ## Alpha Profile ##
    # Generate index arrays
    k_indices = np.arange(Np) - (Np // 2)  # Equivalent to k_eq
    q_indices = np.arange(N) - (N // 2)    # Equivalent to q_eq
    
    # Create 2D arrays for broadcasting
    K_eq, Q_eq = np.meshgrid(k_indices, q_indices, indexing='ij')

    # Compute frequency and alpha matrices
    f_matrix = ((K_eq / (2 * Np)) - (Q_eq / (2 * N)))
    alpha_matrix = (K_eq / Np) + (Q_eq / N)
    print("F_matrix and Alpha_matrix shape", f_matrix.shape, alpha_matrix.shape)
    SCF_conj=np.zeros_like(M_conj)
    SCF_nonconj=np.zeros_like(M_nonconj)

    # Convert (F, Alpha) into valid integer indices for the new grid
    f_indices = np.round((f_matrix - f_matrix.min()) / (f_matrix.max() - f_matrix.min()) * (Np-1)).astype(int)
    alpha_indices = np.round((alpha_matrix - alpha_matrix.min()) / (alpha_matrix.max() - alpha_matrix.min()) * (N-1)).astype(int)
    #print("f_indices, alpha_indices", f_indices.shape, alpha_indices.shape)
    #print("M_conj.shape:", M_conj.shape)
    #print("f_indices.shape:", f_indices.shape, f_indices[0:20])
    #print("alpha_indices.shape:", alpha_indices.shape, alpha_indices[0:20])
    # Perform direct assignment (mapping values)
    SCF_conj[alpha_indices.T, f_indices.T] = M_conj
    SCF_nonconj[alpha_indices.T, f_indices.T] = M_nonconj 
    #pdb.set_trace() 

    return SCF_conj, SCF_nonconj

def adaptive_resample(signal_in, lower_cutoff, upper_cutoff, power=2, max_upsample=4, max_downsample=4):
    """
    Adapts the sampling rate of a signal based on its bandwidth and potential non-linear operations.

    Args:
        signal_in: The input signal (numpy array).
        lower_cutoff: The lower cutoff frequency of the signal (normalized, 0 to 1).
        upper_cutoff: The upper cutoff frequency of the signal (normalized, 0 to 1).
        power: The power to raise the signal to (2 or 4). If None, no power operation is performed.
        max_upsample: The maximum allowed upsampling factor.
        max_downsample: The maximum allowed downsampling factor.

    Returns:
        The resampled signal (numpy array).
    """
    print(lower_cutoff, upper_cutoff)
    sampling_rate = 1.0  # Assume normalized sampling rate of 1
    signal_out = signal_in.copy()

    # 1. Calculate Maximum Frequency After Power Operation
    max_freq_after_power = upper_cutoff *power
    print("Max_freq_after_power",max_freq_after_power)
    # 2. Adaptive Upsampling
    required_upsample = 1
    if max_freq_after_power > sampling_rate / 2: # only upsample if needed
        required_upsample = int(np.ceil((max_freq_after_power * 2) / sampling_rate))
        required_upsample = min(required_upsample, max_upsample) #limit to max upsample factor.'
        print(required_upsample)
        if required_upsample > 1:
            signal_out = signal.resample_poly(signal_out, required_upsample, 1) #upsample using polyphase resampling.
            sampling_rate *= required_upsample
            #Apply Low Pass Filter after upsampling.
            nyquist = sampling_rate / 2.0
            cutoff_norm = upper_cutoff / nyquist #normalized cut-off frequency for the filter
            if cutoff_norm < 1.0: #only filter if needed.
              numtaps = 101 #filter length.
              taps = signal.firwin(numtaps, cutoff_norm, pass_zero='lowpass')
              signal_out = signal.lfilter(taps, 1.0, signal_out)

    
    # 3. Downsampling (if needed)
    if max_freq_after_power > 0.4:
        required_downsample = int(np.ceil((max_freq_after_power * 2) / sampling_rate))
        required_downsample = min(required_downsample, max_downsample) #limit to max downsample.
        if required_downsample > 1:
            # Apply low-pass filter before downsampling
            nyquist = sampling_rate / 2.0
            cutoff_norm = 0.4 / nyquist
            if cutoff_norm < 1.0: #only filter if needed.
              numtaps = 101
              taps = signal.firwin(numtaps, cutoff_norm, pass_zero='lowpass')
              signal_out = signal.lfilter(taps, 1.0, signal_out)
            signal_out = signal.resample_poly(signal_out, 1, required_downsample)
            sampling_rate /= required_downsample

    return signal_out

def plot_max_rho_stem(rho, cyclic_frequencies, title_txt= '', y_label=''):
    """
    Plots the maximum coherence magnitude for each cycle frequency as vertical spikes.

    Args:
        rho: The coherence matrix (numpy array) of shape (num_cyclic_freqs, num_baseband_freqs).
        cyclic_frequencies: Array of cyclic frequencies corresponding to rho rows.
    """
    # Find max coherence for each cyclic frequency
    max_rho_values = np.max(rho, axis=1)  # Max along baseband frequency axis
    print("Max_rho_values_length:", len(max_rho_values))
    # Create the stem plot
    plt.figure(figsize=(8, 4),dpi=100)
    plt.stem(cyclic_frequencies, max_rho_values, linefmt='m-', markerfmt='mo', basefmt="k", label="Max SCF per Cycle Freq")

    # Example: Overlay SSCA Output Grid (Replace with actual values if available)
    # ssca_output_cf_grid = np.linspace(cyclic_frequencies.min(), cyclic_frequencies.max(), num=10)
    # plt.scatter(ssca_output_cf_grid, np.zeros_like(ssca_output_cf_grid), color='r', marker='x', label="SSCA Output CF Grid", s=100)

    plt.xlabel("Cycle Frequency (normalized)")
    plt.ylabel(y_label)
    plt.title(title_txt)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()    




def main():
    data_sample_path = '/home/mohitsharma/Desktop/Codes/MC/Source/signal_4000.tim'
    #data_sample_path = '/home/wmi-nuc-m/Documents/Codes/MC/Source/signal_1.tim'
    #data_sample_path ='/home/wmi-nuc-m/Documents/Codes/MC/Data/ml.2022r2/raw/signal_1.tim'
    #data_sample_path ='/home/wmi-nuc-m/Documents/Codes/MC/Data/CSPB2018/CSPB.ML_.2018R1_1/Batch_Dir_1/signal_1.tim'
    data_samples= read_binary(data_sample_path)
    #data_samples=data_samples**2
    noise_floor=3 #(in dB)
    print(data_samples[0:10])
    
    frequencies, power_spectrum = welch(data_samples, window='hamming', nperseg=1024, noverlap=512, detrend=False, return_onesided=False)

    #print("frequencies:", frequencies)
    n=np.arange(len(data_samples))
    power_spectrum_db = 10 * np.log10(power_spectrum)
    shifted_frequencies  = np.fft.fftshift(frequencies)
    print("Shifted frequencies:", len(shifted_frequencies))
    shifted_spectrums = np.fft.fftshift(power_spectrum_db)
    #print("Shifted_spectrum:", shifted_spectrums)
    thresholded_spectrum=shifted_spectrums.copy()
    thresholded_spectrum[thresholded_spectrum>=noise_floor]=-3
    #print("Thresholded_spectrum:", thresholded_spectrum)
    indices = np.where(thresholded_spectrum == -3)[0]
    #print("Indices",indices)
    #print(shifted_frequencies[indices])
    noise_indices = np.where(thresholded_spectrum != -3)[0]
    #print("Noise_indices",noise_indices)
    data_bandwidth=shifted_frequencies[indices][-1]-shifted_frequencies[indices][0]
    center_frequency =(shifted_frequencies[indices][-1]+shifted_frequencies[indices][0])/2
    print("data_bandwidth and center frequency", data_bandwidth, center_frequency)
    noise_spectral_density=np.mean(shifted_spectrums[noise_indices])
    Noise_power = noise_spectral_density*data_bandwidth
    
       
    # # # Plot the power spectral density (PSD)
    # plt.figure()
    # plt.plot(shifted_frequencies, shifted_spectrums)
    # plt.title("Periodogram with shifted frequencies (Power Spectral Density Estimate)")
    # plt.grid(True)
    # plt.xticks(np.arange(-0.5,0.5,0.1))
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Power/Frequency [dB/Hz]")
    # plt.show()

    ######################### FSM-based psd estimate
    # freq, psd = estimate_psd_fsm(data_samples)
    # plt.figure()
    # plt.plot(freq, psd)
    # plt.grid(True)
    # plt.title("Power Spectral Density Estimate using FSM")
    # plt.xticks(np.arange(-0.5,0.5,0.1))
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Power/Frequency [dB/Hz]")
    # plt.show()
    # print("FSM_estimatd_CFO", freq[np.argmax(psd)])

    ##### CFO_Shift_Correction
    #print(data_samples[0:20], data_samples.dtype)
    CFO=shifted_frequencies[np.argmax(shifted_spectrums)]
    CFO_corrected_data=data_samples*np.exp(-1j*2*np.pi*CFO*n)
    
    #resampled_data = adaptive_resample(CFO_corrected_data,shifted_frequencies[indices][0], shifted_frequencies[indices][-1])
    resampled_data=signal.resample_poly(CFO_corrected_data, 1, 1)
    N_psd=len(resampled_data) # number of fft points in PSD estimates using FSM
    SCF_conj, SCF_nonconj=SSCA_Compute(resampled_data)
    cyclic_frequencies = np.linspace(-1, 1, SCF_conj.shape[0])  # Approximate cyclic frequencies
    #print("zero_cyclic frequency location", np.argmin(np.abs(cyclic_frequencies)))
    baseband_frequencies = np.linspace(-0.5, 0.5, SCF_conj.shape[1])
    f,PSD_estimate = estimate_psd_fsm(resampled_data, N_psd)
    plt.figure()
    plt.plot(f, PSD_estimate)
    plt.grid(True)
    plt.title("Power Spectral Density Estimate after resampling using FSM")
    plt.xticks(np.arange(-0.5,0.5,0.1))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [dB/Hz]")
    plt.show()
    rho, rho_conj = Compute_Coherence_function(SCF_nonconj, SCF_conj, PSD_estimate, baseband_frequencies, cyclic_frequencies, N_psd)
    plot_max_rho_stem(rho, cyclic_frequencies, title_txt = 'Non-Conjugate Cycle Frequencies', y_label='Non-conjugate Coherence')
    plot_max_rho_stem(rho_conj, cyclic_frequencies, title_txt='Conjugate Cycle Frequencies', y_label='Conjugate Coherence')
    plot_max_rho_stem(SCF_nonconj, cyclic_frequencies, title_txt = 'Non-Conjugate Cycle Frequencies based on SCF', y_label='SCF_nonconj_magnitude')
    plot_max_rho_stem(SCF_conj, cyclic_frequencies, title_txt='Conjugate Cycle Frequencies based on SCF', y_label='SCF_conj_magnitude')
    
      
        
    
    #baseband_frequencies = np.linspace(-0.5, 0.5, SCF.shape[1])
    center_idx = len(baseband_frequencies) // 2  
    plt.figure(figsize=(10, 6))
    image=plt.imshow(np.abs(SCF_nonconj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_nonconj))/100, origin='lower')
    plt.colorbar(image, label="Nonconjugate SCF Magnitude (dB)")
    plt.xlabel("Baseband Frequency (Hz)")
    plt.ylabel("Cyclic Frequency (Hz)")
    plt.title("Non conjugate Spectral Correlation Function (SCF)")
    
    max_scf_at_cyclic_freq = np.max(SCF_nonconj, axis=1)  # Maximum SCF for each cyclic frequency
    dominant_cyclic_freqs = cyclic_frequencies[max_scf_at_cyclic_freq > (0.5 * np.max(max_scf_at_cyclic_freq))]  # Thresholding

    #print("Dominant Cyclic Frequencies:", dominant_cyclic_freqs)
    plt.show()
    image=plt.imshow(np.abs(SCF_conj), aspect='auto', extent=[baseband_frequencies[0], baseband_frequencies[-1], cyclic_frequencies[0], cyclic_frequencies[-1]], vmax=np.max(np.abs(SCF_conj))/5, origin='lower')
    plt.colorbar(image, label="Conjugate SCF agnitude (dB)")
    plt.xlabel("Baseband Frequency (Hz)")
    plt.ylabel("Cyclic Frequency (Hz)")
    plt.title("Conjugate Spectral Correlation Function (SCF)")
    
    max_scf_at_cyclic_freq = np.max(SCF_nonconj, axis=1)  # Maximum SCF for each cyclic frequency
    dominant_cyclic_freqs = cyclic_frequencies[max_scf_at_cyclic_freq > (0.5 * np.max(max_scf_at_cyclic_freq))]  # Thresholding

 
 
if __name__ == "__main__":
    main()    

