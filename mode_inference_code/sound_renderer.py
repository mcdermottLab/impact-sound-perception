
from os import error
import torch
import torchaudio
import torch.nn.functional as nnfunc
import numpy as np
import numpy.matlib as matlib
import scipy
import math
from xitorch.interpolate import Interp1D
from scipy.fftpack import dct, idct, rfft, irfft, ifft, next_fast_len
import torch_dct as tdct
#Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

### NUMPY -- differentiation not needed

lo_lim_freq = 1.
rendering_log_const = 1e-12

def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def freq_to_ERB(freq):
    return 9.265*np.log(1 + freq/(24.7*9.265))

def ERB_to_freq(ERB):
    return 24.7*9.265*(np.exp(ERB/9.265) - 1)

def freq_to_octave(freq):
    return np.log2(freq)

def octave_to_freq(ve):
    return 2.0**ve

def get_element_gp_times(onset, offset, tstep):
    return [np.round(t, decimals=3) for t in np.arange(start=0,stop=offset,step=tstep) if t >= onset]

def get_element_gp_freqs(audio_sr, steps):

    scale = steps["scale"]
    fstep = steps["f"]

    if scale == "ERB":
        freq_to_scale = freq_to_ERB
    elif scale == "octave":
        freq_to_scale = freq_to_octave

    lo = freq_to_scale(lo_lim_freq)
    hi = freq_to_scale(np.floor(audio_sr/2.) - 1.)

    return np.round(np.arange(start=lo, stop=hi, step=fstep), decimals=3)

## RAMPS
# Smooth onsets and offsets to reduce artifacts
def hann_ramp(sr, ramp_duration = 0.010):

    #Make ramps
    t = np.arange(start=0, stop=ramp_duration, step=1/sr)
    off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t ))

    return off_ramp

## WINDOWS
# Creates windows in order to segment and differentially alter the excitation signal
def time_win_shape(x):
    return (1 + np.cos(x))/2.0

def make_overlapping_time_windows(filt_len, tstep, sr):
    """

    Input
    -----
    filt_len: len of time dimension of filter array
    tstep (s): duration of one filter element
    sr (Hz): audio sampling rate

    Output
    ------
    win_arr (n_samples_in_scene, n_win):
        overlapping cos-shaped windows for amplitude modulation in time

    """
    win_len = 2*(np.floor(tstep*sr))
    sig_len = (filt_len - 1)*(win_len/2)

    # Check for erroneous inputs
    if (sig_len % (win_len/2) != 0) or (win_len % 2 != 0):
        error("win_len must be even and sig_len must be divisible by half of win_len")

    n_win = int(np.floor(2*sig_len/win_len + 1))
    win_adv_step = int(np.floor(win_len/2))
    last_win_start_idx = int(np.round(sig_len - np.floor(win_len/2)))

    window = time_win_shape(np.linspace(-np.pi, stop=np.pi, num=int(win_len)))
    win_arr = np.zeros((int(sig_len), n_win))
    win_idx_range = np.arange(start=0, stop=last_win_start_idx, step=win_adv_step)
    for curr_win_idx, curr_start_idx in enumerate(win_idx_range, 1):
        curr_lo_idx = curr_start_idx
        curr_hi_idx = curr_start_idx + int(win_len)
        win_arr[curr_lo_idx:curr_hi_idx, curr_win_idx] = window
    #Half windows at the edges
    n = int(np.floor(win_len/2))
    win_arr[0:win_adv_step, 0] = window[-win_adv_step:]
    win_arr[-win_adv_step:,-1] = window[0:win_adv_step]

    return win_arr, int(sig_len)

def freq_win_shape(x):
    return np.cos(x / 2.0)

def make_overlapping_freq_windows(filt_len, steps, audio_sr):
    """Computes array of log spaced cos-shaped windows

    Input
    -----
    filt_len: len of time dimension of filter array
    lowf (Hz), highf (Hz), estep (ERB): defines cutoffs, widths of frequency channels
    tstep (s): duration of one filter element
    audio_sr (Hz): audio sampling rate

    Output
    ------
    win_arr (n_samples_in_scene, n_channels):
        overlapping cos-shaped windows for amplitude modulation in frequency

    """

    n_freqs = int((filt_len - 1)*np.floor(steps["t"]*audio_sr))

    max_freq = audio_sr/2. - 1;

    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    loscale = freq_to_scale(lo_lim_freq)
    hiscale = freq_to_scale(max_freq)
    n_channels = int(len(get_element_gp_freqs(audio_sr, steps)) + 2);
    freqs = np.linspace(0, stop=max_freq, num=n_freqs)

    scale_cutoffs_1D = np.linspace(loscale, stop=hiscale, num=n_channels)
    scale_cutoffs = np.array([[scale_cutoffs_1D[i], scale_cutoffs_1D[i+2]] for i in range(n_channels - 2)]) #50% overlap
    freq_cutoffs = scale_to_freq(scale_cutoffs)

    win_arr = np.zeros((n_freqs, n_channels))
    for curr_channel in range(n_channels - 2):

        curr_lo_freq = freq_cutoffs[curr_channel][0]
        curr_hi_freq = freq_cutoffs[curr_channel][1]
        curr_lo_freq_idx = np.argmax(freqs > curr_lo_freq)
        curr_hi_freq_idx = np.argmin(freqs < curr_hi_freq)

        curr_lo_scale = scale_cutoffs[curr_channel][0]
        curr_hi_scale = scale_cutoffs[curr_channel][1]
        curr_mean_scale = (curr_hi_scale + curr_lo_scale)/2
        scale_bandwidth = curr_hi_scale - curr_lo_scale

        curr_scale = freq_to_scale(freqs[curr_lo_freq_idx:curr_hi_freq_idx])
        normalized_domain = 2*(curr_scale - curr_mean_scale)/scale_bandwidth
        curr_win = freq_win_shape(np.pi*normalized_domain)
        win_arr[curr_lo_freq_idx:curr_hi_freq_idx, curr_channel + 1] = curr_win

    rfc = freq_cutoffs[::-1,:]
    bandwidthsHz = rfc[:,1] - rfc[:,0]
    bandwidthsHz = np.tile(bandwidthsHz[np.newaxis,:],(filt_len,1)) #inversion necessary to line up with the corr_grid, to make sure noise comes out appropriately (eg pink spectrum looks flat on cochleagram)



    return win_arr[:,1:-1], bandwidthsHz, freq_cutoffs


## EXCITATION
def FM_excitation(erbf0, sig_len, steps, sr):
    """
    Generate a frequency modulated tone

    Input
    -----
    erbf0 (vector of ERB elements): specification of frequency in each window
    sig_len (number of samples): overall duration of window vec in samples
    tstep (s): duration of one time window in erbf0 vector
    sr (Hz): audio sampling rate

    Returns
    -------
    1D vector of frequency modulated tone

    """

    tstep = steps["t"]
    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    timepoints = np.linspace(0, stop=(sig_len - 1)/sr, num=int(np.round(sig_len)))
    erb0 = np.interp(timepoints, tstep*np.arange(0,len(erbf0)), erbf0)
    unclamped_f0 = scale_to_freq(erb0)
    f0 = np.clip(unclamped_f0, lo_lim_freq, sr/2. - 1)

    source = np.sin(2*np.pi*np.cumsum(f0*(1./sr)))
    source /= ( (np.mean(source**2))**0.5 + rendering_log_const)

    return source, unclamped_f0, f0

def f0bound_FM_excitation(erbf0, erbf, sig_len, steps, sr, source_colour="pink"):
    """ Generate a frequency modulated tone

    Input
    -----
    erbf0 (vector of ERB elements): specification of frequency in each window
    sig_len (number of samples): overall duration of window vec in samples
    tstep (s): duration of one time window in erbf0 vector
    sr (Hz): audio sampling rate

    Returns
    -------
    1D vector of frequency modulated tone

    """

    tstep = steps["t"]

    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    timepoints = np.linspace(start=0, stop=(sig_len - 1)/sr, num=int(np.round(sig_len)))

    erbfin = np.interp(timepoints, tstep*np.arange(0,len(erbf)), erbf)
    unclamped_f = scale_to_freq(erbfin)
    f = np.clip(unclamped_f, lo_lim_freq, sr/2. - 1)

    erbf0in = np.interp(timepoints, tstep*np.arange(0,len(erbf0)), erbf0)
    unclamped_f0 = scale_to_freq(erbf0in)
    f0 = np.clip(unclamped_f0, lo_lim_freq, sr/2. - 1)

    if source_colour == "pink":
        amplitude_ratio = np.sqrt(f0/f)
    else:
        amplitude_ratio = 1.0


    return source, unclamped_f, f


def f0bound_excitation(erbf0, erbf, sig_len, steps, sr, duration, freq, source_colour="pink"):

    """ Generate a frequency modulated tone

    Input
    -----
    erbf0 (vector of ERB elements): specification of frequency in each window
    sig_len (number of samples): overall duration of window vec in samples
    tstep (s): duration of one time window in erbf0 vector
    sr (Hz): audio sampling rate

    Returns
    -------
    1D vector of frequency modulated tone

    """

    tstep = steps["t"]
    t_vec1 = np.arange(0,duration,1./sr) +1./sr
    t_vec2 = np.arange(0,(sig_len - 1)/sr,1./sr) +1./sr

    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    timepoints = np.linspace(start=0, stop=(sig_len - 1)/sr, num=int(np.round(sig_len)))

    erbfin = np.interp(timepoints, tstep*np.arange(0,len(erbf)), erbf)
    unclamped_f = scale_to_freq(erbfin)
    f = np.clip(unclamped_f, lo_lim_freq, sr/2. - 1)

    erbf0in = np.interp(timepoints, tstep*np.arange(0,len(erbf0)), erbf0)
    unclamped_f0 = scale_to_freq(erbf0in)
    f0 = np.clip(unclamped_f0, lo_lim_freq, sr/2. - 1)

    if source_colour == "pink":
        amplitude_ratio = np.sqrt(f0/f)
    else:
        amplitude_ratio = 1.0


    source = np.zeros(t_vec1.shape[0])
    source += np.sin(2.*np.pi*freq[0]*t_vec1)



    return source, unclamped_f, f

def white_excitation(sig_len, sr):
    """ Generate bandpass white noise
    Input
    -----
    duration (s)
    sr (sampling rate, Hz)

    Returns
    -------
    1D vector of constant amplitude white noise
    """


    nyq_freq = int(np.floor(sr/2))
    hi_lim_freq = nyq_freq - 1
    lo_idx = int(np.ceil(lo_lim_freq/(1. * nyq_freq)*sig_len))
    hi_idx = int(np.floor(hi_lim_freq/(1. * nyq_freq)*sig_len))

    noise_spec = np.zeros((sig_len,))
    noise_spec[lo_idx:hi_idx] = np.random.randn(hi_idx - lo_idx)
    source = idct(noise_spec)
    source /= ( (np.mean(source**2))**0.5 + rendering_log_const)

    return source

def pink_excitation(duration, sr, band=[]):
    """pink
    generates pink noise,
    power spectral density (energy or power per frequency interval)
    is inversely proportional to the frequency of the signal.
    In pink noise, each octave (halving/doubling in frequency)
    carries an equal amount of noise energy.
    Input
    -----
    level (dB rms), relative to ...
    band ([Hz, Hz]):
        low and high limit of bandpass,
        but defaults to not bandpass
    duration (s)
    sr: sampling rate (Hz)
    Returns
    -------
    noise
    """

    #by default, non-zero amplitude spans up to Nyquist frequency
    if len(band) == 0:
        band = [0,int(sr/2.)]

    n = int(np.round(duration*sr))
    binFactor = n/(1.*sr) #move between frequencies and bin indexes
    lowPassBin = int(np.round(band[0]*binFactor))
    highPassBin = int(np.round(band[1]*binFactor))
    x_real = np.zeros(n)
    x_imag = np.zeros(n)
    x_real[lowPassBin:highPassBin+1]=np.random.randn(1+highPassBin-lowPassBin)
    x_imag[lowPassBin:highPassBin+1]=np.random.randn(1+highPassBin-lowPassBin)
    spectrum = x_real + x_imag*1j

    # divide each element of the spectrum by f^(1/2).
    # Since power is the square of amplitude,
    # this means we divide the power at each component by f
    pinkWeights = binFactor*np.arange(1,n+1)
    pinkSpectrum = np.divide(spectrum,np.sqrt(pinkWeights))

    noise = np.real(ifft(pinkSpectrum))
    noise = noise[0:int(duration*sr)]
    noise /= (np.sqrt(np.mean(np.square(noise))) + rendering_log_const )

    return noise

def harmonic_excitation(erbf0, which_harmonics, sig_len, steps, audio_sr,duration):

    tstep = steps["t"]
    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    ##Generating harmonic stack source with frequency modulation
    hi_lim_freq = audio_sr/2. - 1;

    #Sum up the FM harmonics
    f0 = scale_to_freq(erbf0)
    excitation = np.zeros(int(np.round(duration*audio_sr)))
    clamped_f0=[]
    for i in which_harmonics:
        f = f0 * i
        if np.all(f > hi_lim_freq):
            break
        erbf = freq_to_scale(f)
        h, unclamped_f, clamped_f = f0bound_excitation(erbf0,erbf, sig_len, steps, audio_sr,duration,f)
        excitation += h
        if i == 1:
            clamped_f0 = clamped_f
        excitation /= max(excitation)



    return excitation, clamped_f0

## SUBBANDS
def generate_subbands(x, filterbank):
    """generate_subbands
    Generate the subbands of a signal using a filterbank with n_channels. The
    signal in the input Sound object and filterbank must have matching lens.
    Parameters
    ----------
    x : Sound object
        Input sound.
    filterbank : 2-D array of floats
        * Array axis semantics - filter_len x n_subbands
        * Filter bank in frequency domain
        * filter_len should equal signal_len
    Returns
    -------
    subbands : 2-D array of floats
        Array axis semantics - signal_len x n_subbands
        Input signal that is filtered into subbands
    """
    # Check for erroneous input
    sig_len = len(x)
    filt_len, n_channels = filterbank.shape
    if sig_len != filt_len:
        raise ValueError("signal len must equal filter len")

    fft_len = next_fast_len(sig_len)
    fft_sig = rfft(x, fft_len)[:sig_len].reshape(sig_len, 1)
    filtered_subbands = filterbank*matlib.repmat(fft_sig, 1, n_channels)
    subbands = irfft(filtered_subbands, fft_len, axis=0)[:sig_len, :]

    return subbands


### TORCH - differentiation needed

def collapse_subbands(subbands, filterbank):
    """collapse_subbands
    subbands : batch x signal_len x n_subbands
    filterbank : batch x filter_len x n_subbands
    """

    # Check for erroneous input
    sig_len = subbands.shape[1]
    filt_len = filterbank.shape[1]
    if sig_len != filt_len:
        raise ValueError("signal len must equal filter len")

    dct_subbands = tdct.dct(subbands.permute(0,2,1))
    filtered_subbands = filterbank.permute(0,2,1)*dct_subbands
    mod_subbands = tdct.idct(filtered_subbands)

    return torch.sum(mod_subbands, 1) #batch siglen

def modulate_subbands(energy_grid, subbands, win_arr, tstep, sr):
    """modulate_subbands
    energy_grid (batch, n_channels, n_windows)
    subbands(batch, len(x), n_channels)
    win_arr (batch, len(x), n_windows)
    """

    n_batches = subbands.shape[0]
    n_channels = subbands.shape[2]
    n_windows = win_arr.shape[2]
    divisor = torch.full((n_batches, n_channels,n_windows), 2.*np.floor(tstep*sr)).to(subbands.device)
    divisor[:, :, [0,-1]] /= 2.
    tsw =subbands[:, :, :,np.newaxis] * win_arr[:, :, np.newaxis,:] #batch len(x) nchannels nwindows
    scalars = torch.sum(torch.square(tsw),1) #batch nchannels nwindows
    scalars = torch.div(scalars, divisor)
    scalars = torch.sqrt(scalars)
    scalars = torch.div(energy_grid, scalars)
    mod_subbands = tsw * scalars[:,np.newaxis,:,:]
    mod_subbands = torch.sum(mod_subbands,3)

    return mod_subbands

def generate_noise(filt, subbands, win_arr, filterbank, bandwidthsHz, duration, steps, audio_sr, rms_ref):
    """Generate time-varying noise
    Input
    -----
    filt (n_windows, n_channels): 2D GP sampled amplitudes

    Returns
    -------
    1D vector of amplitude modulated noise
    """

    #Amplitude modulate noise
    energy_grid = rms_ref*(10.0**( (filt + 10*torch.log10(bandwidthsHz))/20.0 )) - rendering_log_const
    mod_subbands = modulate_subbands(energy_grid.permute(0, 2, 1), subbands, win_arr, steps["t"], audio_sr)
    AM_noise = collapse_subbands(mod_subbands, filterbank)

    n_samples = int(np.floor(audio_sr*duration))
    start_point = max(0, int(np.floor((len(AM_noise) - n_samples)/2))-1)
    AM_noise = AM_noise[:,start_point:start_point+n_samples]

    return AM_noise

def modulate_harmonic(excitation, filt, win_arr, filterbank, bandwidthsHz, rms_ref,freq,filtbank_freq,sr,duration,spec_slope,expo):


    batch, wlen, n_windows = win_arr.shape
    batch, flen, n_channels = filterbank.shape

    batch = freq.shape[0]
    if wlen == flen:
        sig_len = wlen
    else:
        error("Incorrect array lens")

    half_sr = sr/2;
    mu = 400.0/half_sr
    lambd = 15
    centre = 6000




    t_vec1 = torch.from_numpy(np.arange(0,duration,1./sr) +1./sr).type(torch.FloatTensor)
    output = torch.zeros(batch,t_vec1.shape[0])
    for b in range(0,batch):
        exc_term = - spec_slope*((freq[b,:]/5000)**(expo))
        x = torch.from_numpy(filtbank_freq).type(torch.FloatTensor)
        y = filt[0,0,:]

        norm_filter = Interp1D(x, y, method = "cspline", extrap = "bound")
        new_harm = norm_filter(freq[b,:])
        new_harm = new_harm+exc_term
        new_harm1 = 10**(new_harm/20)



        for indd in range(0,freq.shape[1]):
                output[b, :] = output[b, :] +10*(new_harm1[indd]*torch.sin(2*np.pi*freq[b,indd]*t_vec1)/freq.shape[1])


    return output

def generate_impact(Amp,mode_freq,RT60,win_arr, filterbank, duration, steps, sr):

    batch, wlen, n_windows = win_arr.shape
    batch, flen, n_channels = filterbank.shape

    batch = mode_freq.shape[0]
    if wlen == flen:
        sig_len = wlen
    else:
        error("Incorrect array lens")

    t_vec1 = torch.arange(0,duration+1./sr,1./sr).to(Amp.device)

    
    output1 = torch.zeros(t_vec1.shape[0]).to(Amp.device)
    
    mass = 0.05
    stiff = 1.0*10**8
    duration2 = np.pi*(mass/stiff)**0.5
    if duration2>duration:
        duration = duration2


    t_vec2 = torch.arange(0,duration2+1./sr,1./sr).to(Amp.device)

    force = torch.sin((stiff/mass)**0.5 * t_vec2).to(Amp.device)
    output = torch.zeros(batch,int(t_vec1.shape[0])).to(Amp.device)
    force = force[np.newaxis,np.newaxis,:]


    #size of RT60: [batch, n_channels]
    output1 = 10**(Amp.unsqueeze(2)/20)*10**(-3*t_vec1.unsqueeze(0).unsqueeze(0)/RT60.unsqueeze(2))*torch.sin(2*np.pi*mode_freq.unsqueeze(2)*t_vec1.unsqueeze(0).unsqueeze(0)).to(Amp.device)
    
    # size of output1
    output2 = torch.sum(output1,dim =1)

    #add an initial impulse to output2



    output = torchaudio.functional.conv1d(output2,force,padding = tuple([math.ceil(0.5*(force.shape[2]-1))]))
    #size of output2: [batch, t_vec1.shape[0]]
    #size of output: [batch, t_vec1.shape[0]]

        
    return output    


def generate_IR(Amp,mode_freq,RT60,win_arr, filterbank, duration, steps, sr):

    batch, wlen, n_windows = win_arr.shape
    batch, flen, n_channels = filterbank.shape

    batch = mode_freq.shape[0]
    if wlen == flen:
        sig_len = wlen
    else:
        error("Incorrect array lens")

    t_vec1 = torch.arange(0,duration+1./sr,1./sr).to(Amp.device)

    
    output1 = torch.zeros(t_vec1.shape[0]).to(Amp.device)
    



    output = torch.zeros(batch,int(t_vec1.shape[0])).to(Amp.device)


    #size of RT60: [batch, n_channels]
    output1 = 10**(Amp.unsqueeze(2)/20)*10**(-3*t_vec1.unsqueeze(0).unsqueeze(0)/RT60.unsqueeze(2))*torch.sin(2*np.pi*mode_freq.unsqueeze(2)*t_vec1.unsqueeze(0).unsqueeze(0)).to(Amp.device)

    # size of output1
    output2 = torch.sum(output1,dim =1)

    #add an initial impulse to output2




    #size of output2: [batch, t_vec1.shape[0]]
    output = output2
    #size of output: [batch, t_vec1.shape[0]]

        
    return output    

def add_noise_onset(waveform, noise_bands, amp, decay_rate):
    if onset > waveform.shape[1]:
        return waveform
    noise = noise[:, :waveform.shape[1]]
    noise = noise[:, :onset]
    noise = noise * snr
    waveform[:, :onset] = waveform[:, :onset] + noise
    return waveform
    







def generate_harmonic(excitation, filt, win_arr, filterbank, bandwidthsHz, duration, steps, audio_sr, rms_ref,freq,filtbank_freq,spec_slope,expo):

    output = modulate_harmonic(excitation, filt, win_arr, filterbank, bandwidthsHz, rms_ref,freq,filtbank_freq,audio_sr,duration,spec_slope,expo)
    #Apply ramps
    return output

def register_element(element_wave, element_timing, scene_len, scene_duration, audio_sr):

    onset = element_timing[0];
    offset = min(element_timing[1], scene_duration);
    sample_start = max(0, int(np.floor(onset*audio_sr)));

    pad1 = torch.zeros(1, sample_start).to('cpu')
    pad2 = torch.zeros(1, scene_len.shape[0] - element_wave.shape[0] - sample_start).to('cpu')

    source_wave = torch.cat((pad1[0,:], torch.from_numpy(element_wave), pad2[0,:]), dim=0)

    return source_wave

def excitation_from_latents(obs_len, latents, audio_sr, steps, duration, win_arr_len):
    scene_wave = np.zeros(obs_len)
    all_element_ts = np.array([])
    grouped_element_ts = []
    for element_idx in range(0,latents.shape[1]):
        erbf0 = freq_to_ERB(latents[0, element_idx,0])
        onset = latents[0,element_idx,1]; offset=latents[0,element_idx,2]
        element_length = int(np.floor((offset - onset)*audio_sr))
        element_ts_list = get_element_gp_times(onset, offset, steps["t"])
        grouped_element_ts.append(element_ts_list)
        element_ts = np.array(element_ts_list)
        all_element_ts = np.concatenate((all_element_ts, element_ts))
        erbf0vec = np.full(element_ts.shape, erbf0)
        element_i, clamped_f0 = harmonic_excitation(erbf0vec, np.arange(1,200), element_length, steps, audio_sr,duration)
        scene_wave = torch.from_numpy(element_i)
    delta_len = win_arr_len - len(scene_wave)
    if delta_len % 2 == 0:
        pad = np.zeros((int(delta_len/2),))
        pad = [pad, pad]
    else:
        pad = [np.zeros((int(delta_len/2),)), np.zeros((1+int(delta_len/2),))]
    scene_wave = np.concatenate((pad[0],scene_wave,pad[1]))

    return scene_wave

def make_mask(frame_edges, freq_cutoffs, latents, flen):

    t_mask = []; f_idxs = [];

    for frame_idx, frame in enumerate(frame_edges):

        inside_element = []
        for element_idx in range(latents.shape[1]):
            onset = latents[0,element_idx,1]; offset=latents[0,element_idx,2]
            inside_element.append((frame[0] >= onset) and (frame[1]<=offset))

        if any(inside_element):
            t_mask.append(1.0)
            element_idx = [i for i, x in enumerate(inside_element) if x][0]
            element_frequency = latents[0, element_idx, 0]
            inside_filter = [element_frequency >= fc[0] and element_frequency <= fc[1] for fc in freq_cutoffs]
            filter_idx = [i for i, x in enumerate(inside_filter) if x][0]
            f_idxs.append(max(0,filter_idx-1))
        else:
            t_mask.append(0.0)
            f_idxs.append(0)

    f_idxs = np.array(f_idxs)[:, np.newaxis]
    t_mask = np.array(t_mask)[np.newaxis]
    mask = (np.arange(flen)[np.newaxis] > f_idxs).T * t_mask

    return mask

def edges(thop, num_frames, twin):
    frame_left = thop*np.arange(num_frames)
    frame_right = frame_left + twin
    return [ [frame_left[frame_idx], frame_right[frame_idx]] for frame_idx in range(len(frame_left)) ]
