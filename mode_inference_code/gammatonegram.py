

from __future__ import division
import numpy as np
import scipy.signal as sps
import torch
import torchaudio
import math

def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100,
                    maxfreq=10000, maxlen=1024):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.

    """

    wts = np.zeros([nfilts,nfft])

    #after Slaney's MakeERBFilters
    EarQ = 9.26449; minBW = 24.7; order = 1;

    nFr = np.array(range(nfilts)) + 1
    em = EarQ*minBW
    cfreqs = (maxfreq+em)*np.exp(nFr*(-np.log(maxfreq + em)+np.log(minfreq + em))/nfilts)-em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(nfft/2 + 1)))
    ucirc = np.exp(1j*2*np.pi*ucircArray/nfft);

    ERB = width*np.power(np.power(cfreqs/EarQ,order) + np.power(minBW,order),1/order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B/sr)
    theta = 2*np.pi*cfreqs/sr
    pole = r*np.exp(1j*theta)
    T = 1/sr
    ebt = np.exp(B*T); cpt = 2*cfreqs*np.pi*T
    ccpt = 2*T*np.cos(cpt); scpt = 2*T*np.sin(cpt);
    A11 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2);
    A12 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3+2**1.5)*scpt,ebt),2);
    A13 = -np.divide(np.divide(ccpt,ebt) + np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2);
    A14 = -np.divide(np.divide(ccpt,ebt) - np.divide(np.sqrt(3-2**1.5)*scpt,ebt),2);
    zros = -np.array([A11, A12, A13, A14])/T;
    wIdx = range(int(nfft/2 + 1))
    gain = np.abs((-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) - np.sqrt(3 - 2**(3/2))*  np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 - 2**(3/2)) *  np.sin(2*cfreqs*np.pi*T)))*(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) -  np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) /(-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cfreqs*np.pi*T) +  2*(1 + np.exp(4*1j*cfreqs*np.pi*T))/np.exp(B*T))**4);
    wts[:, wIdx] =  ((T**4)/np.reshape(gain,(nfilts,1))) * np.abs(ucirc-np.reshape(zros[0],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[1],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[2],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[3],(nfilts,1)))*(np.abs(np.power(np.multiply(np.reshape(pole,(nfilts,1))-ucirc,np.conj(np.reshape(pole,(nfilts,1)))-ucirc),-GTord)));
    wts = wts[:,range(maxlen)];

    return wts, cfreqs


def gtg_settings(sr=44100,twin=0.025,thop=0.010,N=64,
                  fmin=1,fmax=22050,width=1.0):

    # Calculate a spectrogram-like time frequency magnitude array
    # based on Gammatone subband filters.  Waveform X (at sample
    # rate SR) is passed through an N (default 64) channel gammatone
    # auditory model filterbank, with lowest frequency FMIN (50)
    # and highest frequency FMAX (SR/2).  The outputs of each band
    # then have their energy integrated over windows of TWIN secs
    # (0.025), advancing by THOP secs (0.010) for successive
    # columns.  These magnitudes are returned as an N-row
    # nonnegative real matrix, Y.
    # WIDTH (default 1.0) is how to scale bandwidth of filters
    # relative to ERB default (for fast method only).
    # F returns the center frequencies in Hz of each row of Y
    # (uniformly spaced on a Bark scale).
    nfft = int(2**(np.ceil(np.log(2*twin*sr)/np.log(2))))
    nhop = int(np.round(thop*sr))
    nwin = int(np.round(twin*sr))
    [gtm,gtg_center_freqs] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft/2+1))

    print("nfft:  ",nfft)

    #pytorch stft is based on librosa
    #https://librosa.org/doc/latest/_modules/librosa/core/convert.html#fft_frequencies
    spectrogram_freqs = np.linspace(0, float(sr) / 2, int(1 + nfft // 2), endpoint=True)

    gtm_dB = 20*np.log10(gtm)
    thresh = gtm_dB > -3
    gtg_filters_range = []
    for i in range(gtm_dB.shape[0]):
        idxs = [i for i,x in enumerate(thresh[i,:]) if x]
        lower_idx = idxs[0]; higher_idx = idxs[-1] + 1
        gtg_filter_range = [spectrogram_freqs[lower_idx], spectrogram_freqs[higher_idx]]
        gtg_filters_range.append(gtg_filter_range)

    return nfft, nhop, nwin, gtm, gtg_filters_range

def cosine_filterbank(num_filters, sampling_freq, filter_len):
    # Compute the time axis
    filter_order = filter_len - 1
    f_c = np.linspace(0, sampling_freq/2, num_filters+2)[1:-1] # center frequencies
    B = 1000 # bandwidth in Hz

    # Create window function
    w = torch.hamming_window(filter_len)

    # Create filterbank
    filterbank = torch.zeros(num_filters, filter_len)

    for i in range(num_filters):
        # Compute impulse response
        n = torch.arange(filter_len)
        h = 2 * B * torch.cos(2 * np.pi * f_c[i] * n / sampling_freq) * w
        # Normalize impulse response
        h = h / torch.sqrt(torch.sum(h**2))
        # Shift impulse response to center
        h = torch.roll(h, filter_order // 2)
        # Save filter to filterbank
        filterbank[i, :] = h
    
    return filterbank

def gammatone_filterbank(x, sample_rate, n_filters=10, f_min=20, f_max=None):
    """
    Implements a time-domain gammatone filterbank in PyTorch.
    
    Arguments:
        x (Tensor): Input waveform of shape (batch_size, num_samples).
        sample_rate (int): Sampling rate of the input waveform in Hz.
        n_filters (int): Number of filters in the filterbank.
        f_min (float): Minimum frequency of the filterbank in Hz.
        f_max (float): Maximum frequency of the filterbank in Hz. If None, set to half the sample rate.
    
    Returns:
        Tensor of shape (batch_size, n_filters, num_samples) containing the filterbank outputs for each input waveform.
    """
    if f_max is None:
        f_max = sample_rate / 2.0
    
    # Calculate the center frequencies of the filters.
    erb_min = 24.7 * (4.37 * f_min / 1000.0 + 1)
    erb_max = 24.7 * (4.37 * f_max / 1000.0 + 1)
    erbs = torch.linspace(erb_min, erb_max, n_filters)
    fc = (erbs / 24.7 - 1) / 4.37 * 1000.0
    
    # Calculate the filter coefficients.
    t = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device) / sample_rate
    b = 1.019 * 2 * math.pi * 24.7 * (4.37 * fc / 1000.0 + 1) / sample_rate
    a0 = 1.0
    a2 = 0.0
    b0 = 1.0
    b1 = -2 * torch.cos(b)
    b2 = 1.0
    
    # Apply the filters to the input waveform.
    output = torch.zeros((x.shape[0], n_filters, x.shape[-1]), dtype=torch.float32, device=x.device)
    for i in range(n_filters):
        y = torch.zeros_like(x)
        for j in range(len(b)):
            if j == 0:
                y = b0 / a0 * x
            elif j == 1:
                y = (b0 / a0 * x * a0 + b1 / a0 * y[:-1] * a2 - b1 / a0 * x[1:] * a2) / b0
            else:
                y = (b0 / a0 * x * a0 + b1 / a0 * y[:-1] * a2 - b1 / a0 * x[1:] * a2 - b2 / a0 * y[:-2] * a0) / b0
        output[:, i, :] = y
    
    return output

def gammatone_filter_bank(signal, sample_rate, num_channels, f_min, f_max):
    # Compute filter parameters
    duration = signal.shape[1] / sample_rate
    t = np.arange(0,duration+1./sample_rate,1./sample_rate)
    f_k = np.linspace(f_min, f_max, num_channels)
    b = 1.019 * 2 * np.pi * 24.7 * (4.37 * f_k / 1000 + 1)

    # Compute filter coefficients
    a0 = t
    a2 = 0
    b1 = -2*np.exp(-b*t)*np.cos(2*f_k*np.pi*t)
    b2 = np.exp(-2*b*t)
    coeffs = np.column_stack([b1+b2, -2*b2, np.ones_like(b2)])

    # Apply filters
    signal = torch.Tensor(signal)
    output = []
    #size of signal is (num_channels, signal_length)
    #size of coeffs is (num_channels, 3)
    for i in range(num_channels):
        #assemble filtered singal by convolution of signal and filter coefficients
        filtered = torch.nn.functional.conv1d(signal,
                                                torch.Tensor(coeffs[i, :, None]),
                                                padding=b1.size//2)
        filtered = torch.nn.functional.conv1d(signal, 
                                              torch.Tensor(coeffs[i, :, None]), 
                                              padding=b1.size//2)
        output.append(filtered.squeeze())
    return output


def gammatonegram(x, nfft, nhop, nwin, gtm, rms_ref, window, log_constant=1e-7, dB_threshold=-60):
    """
    # Sat May 27 15:37:50 2017 Maddie Cusimano mcusi@mit.edu, converted to python
    """

    # perform FFT and weighting in amplitude domain
    # note: in MATLAB, abs(spectrogram(X, hanning(nwin), nwin-nhop, nfft, SR))
    #                  = abs(specgram(X,nfft,SR,nwin,nwin-nhop))
    # in python approx = sps.spectrogram(x, fs=sr, window='hann', nperseg=nwin, 
    #                    noverlap=nwin-nhop, nfft=nfft, detrend=False, 
    #                    scaling='density', mode='magnitude')



    Sxx = torchaudio.functional.spectrogram(x, pad=0, window=window, n_fft=nfft, hop_length=nhop, win_length=nwin, power=2.0, normalized=True) #size; (batch, freq, time)
    y = torch.abs(Sxx) / (nfft*rms_ref)
    y = torch.where(y < log_constant, y + log_constant, y)
    y = torch.clamp(y, min=log_constant)
    y = 20.*torch.log10(y) #convert to dB
    y = torch.clamp(y, min=dB_threshold)

    return y

def gammatonegram2(x, nfft,rms_ref,window=torch.hann_window(1024), log_constant=1e-7, dB_threshold=-60):
    
    nhop = int(nfft/2)
    nwin = nfft

    
    Sxx = torchaudio.functional.spectrogram(x, pad=0, window=window, n_fft=nfft, hop_length=nhop, win_length=nwin, power=1.0, normalized=True) #size; (batch, freq, time)
    y = torch.abs(Sxx) / (nfft*rms_ref)
    y = torch.where(y < log_constant, y + log_constant, y)
    y = torch.clamp(y, min=log_constant)
    y = 20.*torch.log10(y) #convert to dB
    y = torch.clamp(y, min=dB_threshold)

    return y
