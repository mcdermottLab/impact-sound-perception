#Arrays
import torch
import torchaudio
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.fft
import numpy as np
import math
import numpy
import scipy
from scipy.signal import lfilter
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import time_domain_cochleagram as tdc
import chcochleagram.downsampling as downsampling
#Custom code
import sound_renderer as dr
import gammatonegram as dg
from A_weighting import A_weighting
#Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdb



#Settings
dtype = torch.FloatTensor 
steps = {"scale":"ERB", "t":0.05, "f":0.75}
steps2 = {"scale":"ERB", "t":0.05, "f":4.0}
rms_ref = 2*10**-6

softplus = nn.Softplus()
MSE_loss = nn.MSELoss()
L1_loss = nn.L1Loss()

class ImpactSound(nn.Module):
    def __init__(self, observed_sound, audio_sr, regularization_weight=1e-4):
        super().__init__()

        """
        observed_sound: array of floats, soundwave
        audio_sr: integer, audio sampling rate
        latents: contains information to generate periodic excitation

        Note: this scene does not have priors!
        """

        print("Defining dimensions...")
        self.n_sounds = observed_sound.shape[0]
        self.steps = steps2
        self.audio_sr = audio_sr
        self.rms_ref = rms_ref
        self.duration = 1.0*observed_sound.shape[1]/self.audio_sr
        self.observed_sound = torch.Tensor(observed_sound).type(dtype)

 

        self.t = dr.get_element_gp_times(0.0, self.duration, self.steps["t"])
        self.dt = len(self.t)
        self.f = dr.get_element_gp_freqs(self.audio_sr, self.steps)
        self.df = len(self.f)

        print('f:' ,self.n_sounds)
       
        print("Pre-compute rendering components...")
        win_arr, self.sig_len = dr.make_overlapping_time_windows(self.dt, self.steps["t"], self.audio_sr)
        self.win_arr = torch.Tensor(win_arr[np.newaxis, :, :]).type(dtype)
        filterbank, bandwidthsHz, filt_freq_cutoffs = dr.make_overlapping_freq_windows(self.dt, self.steps, self.audio_sr)
        self.filterbank = torch.Tensor(filterbank[np.newaxis,:,:]).type(dtype)
        self.bandwidthsHz = torch.Tensor(bandwidthsHz[np.newaxis, :, :]).type(dtype)

        print("self.filterbank", self.filterbank.shape)
        
        print("Generate periodic excitation...")
        
        
    


        filtbank_freq = (filt_freq_cutoffs[:,0] + filt_freq_cutoffs[:,1])/2
        self.filtbank_freq = filtbank_freq




        print("Defining latent filter...")
        b, a = A_weighting(audio_sr)
        num_modes = 5
        fin_freq_all = np.zeros([self.n_sounds,num_modes])
        RT60_s = np.zeros([self.n_sounds,num_modes])

        vals = np.zeros([self.n_sounds,num_modes])

        ##Filter parameter
        for indd in range(0,self.n_sounds):
            obs_sounds = torch.Tensor(observed_sound).type(dtype)
            y = lfilter(b, a, observed_sound[indd])
            obs_fft = scipy.fft.rfft(y)
            obs_fft = abs(obs_fft)
            obs_fft = np.clip(obs_fft,1e-4,50)
            # apply a moving average filter on the power spectrum
            obs_fft1 = np.convolve(obs_fft, np.ones(10), 'same') / 10
            #convert obs_fft to dB
            
            df = observed_sound.shape[1]/len(obs_fft)
            freq = np.arange(0,observed_sound.shape[1],df)
            peaks, properties = scipy.signal.find_peaks(20*np.log10(obs_fft1),distance =100) 
            peaks = peaks[np.where(20*np.log10(obs_fft1[peaks]) > -30)]

            if peaks.shape[0] < num_modes:
                peaks, properties = scipy.signal.find_peaks(20*np.log10(obs_fft1),distance =50)
                peaks = peaks[np.where(20*np.log10(obs_fft1[peaks]) > -30)]
            
            
            pow =observed_sound[indd]**2
            
           
            smooth_pow = 2*np.convolve(pow, np.ones(441), 'same') / 441
            pow_dB =10*np.log10(abs(smooth_pow+1e-8))
            index1 = np.where(pow_dB < -60)[0]
            if index1.shape[0] == 0:
                index1 = np.where(pow_dB < -40)[0]
            index = index1[0]
            if index<1000:
                index = np.min([1000 , index1[np.where(index1>1000)[0][0]]])
            RT60_s[indd] = index/self.audio_sr *np.ones(num_modes)


            if peaks.shape[0] < num_modes:
                peaks = np.concatenate((peaks, np.zeros(num_modes-peaks.shape[0])))
            peaks = peaks.astype(int)
            arg_peaks = np.argsort(obs_fft[peaks])
            sort_peaks = peaks[arg_peaks[-num_modes:]] *df
            fin_freq = sort_peaks
            fin_freq_all[indd] = fin_freq
 
            fin_freq1 = peaks[arg_peaks[-num_modes:]]
            fin_freq1 = fin_freq1.astype(int)
            vals[indd] = 20*np.log10(obs_fft[peaks[arg_peaks[-num_modes:]]])

            


       


        self.mode_freq = nn.Parameter(torch.tensor(fin_freq_all).type(dtype))
        self.Amp = nn.Parameter(torch.tensor(vals-np.max(vals)-np.random.rand()*5).type(dtype))

        RT60 = (torch.tensor(abs(RT60_s))*(0.4*torch.rand(self.mode_freq.shape)+0.9)).type(dtype)
        self.RT60 = nn.Parameter(RT60)
        



       

        print("Defining loss parameters...")
        self.regularization_weight = regularization_weight
        self.eps_sigma = 1.0*torch.ones(self.n_sounds,1).type(dtype) 

        print("Creating gammatonegram components...")
        ##Likelihood parameters
        twin=0.015; thop=0.005;

        self.nfft, self.nhop, self.nwin, gtm, gtg_freq_cutoffs = dg.gtg_settings(twin=twin,thop=thop)
        self.gtm = torch.Tensor(gtm[np.newaxis,:,:]).type(dtype)
        for bb in range(0,self.n_sounds):
            observed_sound[bb] = 0.5*observed_sound[bb]/ np.sqrt(np.amax(observed_sound[bb]**2))
        
        self.window = torch.hann_window(self.nfft).type(dtype)
        self.obs = dg.gammatonegram2(obs_sounds, self.nfft,self.rms_ref,window = self.window)
        nfft1 = int(self.nfft/4)
        self.window2 = torch.hann_window(nfft1).type(dtype)
        self.obs2 = dg.gammatonegram2(obs_sounds, nfft1,self.rms_ref,window = self.window2) 
        nfft1 = int(self.nfft/4)
        self.window3 = torch.hann_window(nfft1).type(dtype)
        self.obs3 = dg.gammatonegram2(obs_sounds,  nfft1,self.rms_ref,window = self.window3)
        nfft1 = int(self.nfft/4)
        self.window4 = torch.hann_window(nfft1).type(dtype)
        self.obs4 = dg.gammatonegram2(obs_sounds,  nfft1,self.rms_ref,window = self.window4)
        
        self.obs_rfft = 20*torch.log10(abs(torch.fft.rfft(obs_sounds))+1e-12)
        self.init_click = obs_sounds[:,0:50]
        # Define the convolution kernel
        kernel_size = 2
        kernel = torch.ones(kernel_size) / kernel_size
        # size of kernel 
        #reshape kernel
        kernel_n = kernel.unsqueeze(0).unsqueeze(0).repeat(1,self.n_sounds,1).type(dtype)
        #size of kernel_n = [1, n_sounds, kernel_size]
        #size of self.obs_rfft = [n_sounds, n_freq, n_time]
        self.obs_rfft = F.conv1d(self.obs_rfft, kernel_n)
        filter_params = {
            'sr': 44100,
            'nfft': self.nfft,
            'n_channels': num_modes+20,
            'low_lim': 30,
            'high_lim': 16000
        }

        
    
        self.filter_params = filter_params
        #size of self.obs = [n_sounds, n_freq, n_time]

        
        
        self.amplitude_threshold = 1e-8 #necessary for gammatonegram stability
        self.coch = tdc.TimeDomainCochleagram(self.filter_params,downsampling,on_gpu=False).to(self.Amp.device)
        self.noise_1 = torch.randn(observed_sound.shape).type(dtype)
        self.noise_1 = self.noise_1/torch.max(torch.abs(self.noise_1))
        self.noise_coch = self.coch.forward(self.noise_1)
        alpha_st = self.coch.forward(self.observed_sound)
        init_est = torch.mean(2*alpha_st[:,:,:200]**2,2) 
        init_est = 10*torch.log10(init_est+1e-12) +30

        self.alpha = nn.Parameter(init_est.type(dtype))
        self.gamma= nn.Parameter((0.05*torch.rand(self.alpha.shape)+0.04).type(dtype))

       



        print("Creating mask for likelihood computation...")

    def plot_step(self):
        return

    def forward(self, step_idx):

        ###Rendering

        x = dr.generate_IR(self.Amp,self.mode_freq,self.RT60,self.win_arr,self.filterbank,self.duration,self.steps,self.audio_sr)
        
        t_vec1 = torch.arange(0,self.duration,1./self.audio_sr).to(x.device)
        noise_coch1 = self.noise_coch*10**(self.alpha.unsqueeze(-1)/20)*10**(-3*t_vec1/self.gamma.unsqueeze(-1))
        noise_filt = torch.sum(noise_coch1,1)
        # add a zero at the end of noise_filt to make it the same size as x
        noise_filt = torch.cat((noise_filt,torch.zeros([noise_filt.shape[0],1]).type(dtype)),1)
        x = x+ noise_filt
        

        #size of x = [n_sounds, n_time]
        

       
        
        ###Likelihood
        # Calculate the rfft of x and find the loss term wrt to the observed sound
        x_rfft = 20*torch.log10(abs(torch.fft.rfft(x))+1e-12)
        # Define the convolution kernel
        kernel_size = 2
        kernel = torch.ones(kernel_size) / kernel_size
        kernel_n = kernel.unsqueeze(0).unsqueeze(0).repeat(1,self.n_sounds,1).type(dtype)

        x_rfft= F.conv1d(x_rfft, kernel_n)
        #size of x_rfft 


        gtg = dg.gammatonegram2(x, self.nfft,self.rms_ref,window = self.window)

        nfft1 = int(self.nfft/4)
        gtg2 = dg.gammatonegram2(x, nfft1,self.rms_ref,window = self.window2)
        nfft1 = int(self.nfft/4)
        gtg3 = dg.gammatonegram2(x,  nfft1,self.rms_ref,window = self.window3)
        nfft1 = int(self.nfft/4)
        gtg4 = dg.gammatonegram2(x,  nfft1,self.rms_ref,window = self.window4)

        



        weight = 0.1
        loss_type = nn.HuberLoss(reduction = 'mean', delta=3.0)
        loss = loss_type(gtg.flatten(),self.obs.flatten())+loss_type(gtg2.flatten(),self.obs2.flatten())+loss_type(gtg3.flatten(),self.obs3.flatten())+loss_type(gtg4.flatten(),self.obs4.flatten())
        loss = loss +weight*loss_type(x_rfft.flatten(),self.obs_rfft.flatten())
        #size of x_rfft is 1, 2, 6000 and self.obs_rfft is 1, 2, 6000  
        return loss, gtg, x



class ImpactSound_sin(nn.Module):
    def __init__(self, observed_sound, audio_sr, regularization_weight=1e-4):
        super().__init__()

        """
        observed_sound: array of floats, soundwave
        audio_sr: integer, audio sampling rate
        latents: contains information to generate periodic excitation

        Note: this scene does not have priors!
        """

        print("Defining dimensions...")
        self.n_sounds = observed_sound.shape[0]
        self.steps = steps2
        self.audio_sr = audio_sr
        self.rms_ref = rms_ref
        self.duration = 1.0*observed_sound.shape[1]/self.audio_sr
        self.observed_sound = torch.Tensor(observed_sound).type(dtype)

 

        self.t = dr.get_element_gp_times(0.0, self.duration, self.steps["t"])
        self.dt = len(self.t)
        self.f = dr.get_element_gp_freqs(self.audio_sr, self.steps)
        self.df = len(self.f)

        print('f:' ,self.n_sounds)
       
        print("Pre-compute rendering components...")
        win_arr, self.sig_len = dr.make_overlapping_time_windows(self.dt, self.steps["t"], self.audio_sr)
        self.win_arr = torch.Tensor(win_arr[np.newaxis, :, :]).type(dtype)
        filterbank, bandwidthsHz, filt_freq_cutoffs = dr.make_overlapping_freq_windows(self.dt, self.steps, self.audio_sr)
        self.filterbank = torch.Tensor(filterbank[np.newaxis,:,:]).type(dtype)
        self.bandwidthsHz = torch.Tensor(bandwidthsHz[np.newaxis, :, :]).type(dtype)

        print("self.filterbank", self.filterbank.shape)
        
        print("Generate periodic excitation...")
        
        
    


        filtbank_freq = (filt_freq_cutoffs[:,0] + filt_freq_cutoffs[:,1])/2
        self.filtbank_freq = filtbank_freq




        print("Defining latent filter...")
        b, a = A_weighting(audio_sr)
        num_modes = 5
        fin_freq_all = np.zeros([self.n_sounds,num_modes])
        RT60_s = np.zeros([self.n_sounds,num_modes])

        vals = np.zeros([self.n_sounds,num_modes])

        ##Filter parameter
        for indd in range(0,self.n_sounds):
            obs_sounds = torch.Tensor(observed_sound).type(dtype)
            y = lfilter(b, a, observed_sound[indd])
            obs_fft = scipy.fft.rfft(y)
            obs_fft = abs(obs_fft)
            obs_fft = np.clip(obs_fft,1e-4,50)
            # apply a moving average filter on the power spectrum
            obs_fft1 = np.convolve(obs_fft, np.ones(10), 'same') / 10
            #convert obs_fft to dB
            
            df = observed_sound.shape[1]/len(obs_fft)
            freq = np.arange(0,observed_sound.shape[1],df)
            peaks, properties = scipy.signal.find_peaks(20*np.log10(obs_fft1),distance =100) 
            peaks = peaks[np.where(20*np.log10(obs_fft1[peaks]) > -30)]

            if peaks.shape[0] < num_modes:
                peaks, properties = scipy.signal.find_peaks(20*np.log10(obs_fft1),distance =50)
                peaks = peaks[np.where(20*np.log10(obs_fft1[peaks]) > -30)]
            
            
            pow =observed_sound[indd]**2
            
           
            smooth_pow = 2*np.convolve(pow, np.ones(441), 'same') / 441
            pow_dB =10*np.log10(abs(smooth_pow+1e-8))
            index1 = np.where(pow_dB < -60)[0]
            if index1.shape[0] == 0:
                index1 = np.where(pow_dB < -40)[0]
            index = index1[0]
            if index<1000:
                index = np.min([1000 , index1[np.where(index1>1000)[0][0]]])
            RT60_s[indd] = index/self.audio_sr *np.ones(num_modes)


            if peaks.shape[0] < num_modes:
                peaks = np.concatenate((peaks, np.zeros(num_modes-peaks.shape[0])))
            peaks = peaks.astype(int)
            arg_peaks = np.argsort(obs_fft[peaks])
            sort_peaks = peaks[arg_peaks[-num_modes:]] *df
            fin_freq = sort_peaks
            fin_freq_all[indd] = fin_freq
 
            fin_freq1 = peaks[arg_peaks[-num_modes:]]
            fin_freq1 = fin_freq1.astype(int)
            vals[indd] = 20*np.log10(obs_fft[peaks[arg_peaks[-num_modes:]]])

            


       


        self.mode_freq = nn.Parameter(torch.tensor(fin_freq_all).type(dtype))
        self.Amp = nn.Parameter(torch.tensor(vals-np.max(vals)-np.random.rand()*5).type(dtype))

        RT60 = (torch.tensor(abs(RT60_s))*(0.4*torch.rand(self.mode_freq.shape)+0.9)).type(dtype)
        self.RT60 = nn.Parameter(RT60)
        



       

        print("Defining loss parameters...")
        self.regularization_weight = regularization_weight
        self.eps_sigma = 1.0*torch.ones(self.n_sounds,1).type(dtype) 

        print("Creating gammatonegram components...")
        ##Likelihood parameters
        twin=0.015; thop=0.005;

        self.nfft, self.nhop, self.nwin, gtm, gtg_freq_cutoffs = dg.gtg_settings(twin=twin,thop=thop)
        self.gtm = torch.Tensor(gtm[np.newaxis,:,:]).type(dtype)
        for bb in range(0,self.n_sounds):
            observed_sound[bb] = 0.5*observed_sound[bb]/ np.sqrt(np.amax(observed_sound[bb]**2))
        
        self.window = torch.hann_window(self.nfft).type(dtype)
        self.obs = dg.gammatonegram2(obs_sounds, self.nfft,self.rms_ref,window = self.window)
        nfft1 = int(self.nfft/4)
        self.window2 = torch.hann_window(nfft1).type(dtype)
        self.obs2 = dg.gammatonegram2(obs_sounds, nfft1,self.rms_ref,window = self.window2) 
        nfft1 = int(self.nfft/4)
        self.window3 = torch.hann_window(nfft1).type(dtype)
        self.obs3 = dg.gammatonegram2(obs_sounds,  nfft1,self.rms_ref,window = self.window3)
        nfft1 = int(self.nfft/4)
        self.window4 = torch.hann_window(nfft1).type(dtype)
        self.obs4 = dg.gammatonegram2(obs_sounds,  nfft1,self.rms_ref,window = self.window4)
        
        self.obs_rfft = 20*torch.log10(abs(torch.fft.rfft(obs_sounds))+1e-12)
        self.init_click = obs_sounds[:,0:50]
        # Define the convolution kernel
        kernel_size = 2
        kernel = torch.ones(kernel_size) / kernel_size
        # size of kernel 
        #reshape kernel
        kernel_n = kernel.unsqueeze(0).unsqueeze(0).repeat(1,self.n_sounds,1).type(dtype)
        #size of kernel_n = [1, n_sounds, kernel_size]
        #size of self.obs_rfft = [n_sounds, n_freq, n_time]
        self.obs_rfft = F.conv1d(self.obs_rfft, kernel_n)
        filter_params = {
            'sr': 44100,
            'nfft': self.nfft,
            'n_channels': num_modes+20,
            'low_lim': 30,
            'high_lim': 16000
        }

        
    
        self.filter_params = filter_params
        #size of self.obs = [n_sounds, n_freq, n_time]

        
        
        self.amplitude_threshold = 1e-8 #necessary for gammatonegram stability
        self.coch = tdc.TimeDomainCochleagram(self.filter_params,downsampling,on_gpu=False).to(self.Amp.device)
        self.noise_1 = torch.randn(observed_sound.shape).type(dtype)
        self.noise_1 = self.noise_1/torch.max(torch.abs(self.noise_1))
        self.noise_coch = self.coch.forward(self.noise_1)
        alpha_st = self.coch.forward(self.observed_sound)
        init_est = torch.mean(2*alpha_st[:,:,:200]**2,2) 
        init_est = 10*torch.log10(init_est+1e-12) +30


       



        print("Creating mask for likelihood computation...")

    def plot_step(self):
        return

    def forward(self, step_idx):

        ###Rendering

        x = dr.generate_IR(self.Amp,self.mode_freq,self.RT60,self.win_arr,self.filterbank,self.duration,self.steps,self.audio_sr)
        
        t_vec1 = torch.arange(0,self.duration,1./self.audio_sr).to(x.device)
        

        #size of x = [n_sounds, n_time]
        

       
        
        ###Likelihood
        # Calculate the rfft of x and find the loss term wrt to the observed sound
        x_rfft = 20*torch.log10(abs(torch.fft.rfft(x))+1e-12)
        # Define the convolution kernel
        kernel_size = 2
        kernel = torch.ones(kernel_size) / kernel_size
        kernel_n = kernel.unsqueeze(0).unsqueeze(0).repeat(1,self.n_sounds,1).type(dtype)

        x_rfft= F.conv1d(x_rfft, kernel_n)
        #size of x_rfft 


        gtg = dg.gammatonegram2(x, self.nfft,self.rms_ref,window = self.window)

        nfft1 = int(self.nfft/4)
        gtg2 = dg.gammatonegram2(x, nfft1,self.rms_ref,window = self.window2)
        nfft1 = int(self.nfft/4)
        gtg3 = dg.gammatonegram2(x,  nfft1,self.rms_ref,window = self.window3)
        nfft1 = int(self.nfft/4)
        gtg4 = dg.gammatonegram2(x,  nfft1,self.rms_ref,window = self.window4)

        



        weight = 0.1
        loss_type = nn.HuberLoss(reduction = 'mean', delta=3.0)
        loss = loss_type(gtg.flatten(),self.obs.flatten())+loss_type(gtg2.flatten(),self.obs2.flatten())+loss_type(gtg3.flatten(),self.obs3.flatten())+loss_type(gtg4.flatten(),self.obs4.flatten())
        loss = loss +weight*loss_type(x_rfft.flatten(),self.obs_rfft.flatten())
        #size of x_rfft is 1, 2, 6000 and self.obs_rfft is 1, 2, 6000  
        return loss, gtg, x





class KnownInstrument(nn.Module):

    def __init__(self, fundamental_frequency, amplitude, observed_sounds):
        super().__init__()

        #Observations -- take the first second of each sound
        #Known fundamnetal frequency
        #Same spectrum
        #Different amounts of effort

        self.n_sounds = len(fundamental_frequency)
        self.n_freq_bins = 10 

        ## Excitation (different across different sound examples)
        #Observe the fundamental
        self.fundamental_frequency = torch.Tensor(fundamental_frequency).type(dtype) #(n_sounds, 1,)
        self.amplitude = torch.Tensor(amplitude).type(dtype) #(n_sounds, 1,)
        #Spectral centroid of the excitation
        self.spectral_centroid = nn.Parameter(torch.full(self.amplitude.shape, 1500.).type(dtype)) #(n_sounds, 1,)

        ## Filter (shared across all sound examples)
        self.spectrum_mu = nn.Parameter(torch.zeros(1,self.n_freq_bins,).type(dtype))
        self.spectrum_sigma = 1.0
        self.spectrums = nn.Parameter(torch.zeros(self.n_sounds, self.n_freq_bins).type(dtype)) #shape: (n_sounds, n_freq_bins)

        ## Observations
        # observed_sounds: (n_sounds, len_sound,)
        self.nfft, self.nhop, self.nwin, gtm, gtg_freq_cutoffs = dg.gtg_settings()
        self.gtm = torch.Tensor(gtm[np.newaxis,:,:]).type(dtype);
        self.observed_sounds = torch.Tensor(observed_sounds).type(dtype)
        self.window = torch.hann_window(self.nwin).type(dtype)
        #observed_gammatonegrams: (n_sounds, t, f)
        self.observed_gammatonegrams = dg.gammatonegram(self.observed_sounds, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)
        self.gtg_sigma = torch.ones((n_sounds,)).type(dtype)

    def prior(self):
        #self.spectrums ~ Normal(self.spectrum_mu, self.spectrum_sigma)
        m = self.spectrum_mu.extend(self.n_sounds, 1)
        log_prior = Normal(m.flatten(), self.spectrum_sigma).log_prob(self.spectrums)
        return log_prior

    def forward(self):

        ## Compute the prior probability
        log_prior = self.prior()


        ## Compute the log_likelihood (-loss)
        gtg = dg.gammatonegram(x, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)
        log_likelihood = Normal(gtg.flatten(), self.gtg_sigma).log_prob(self.obs.flatten()).sum()
        loss = -log_prior - log_likelihood

        return loss




class SquaredExponential(nn.Module):
    def __init__(self, xpoints, hyperpriors, n_sounds=1):
        super().__init__()

        ## GP Hyperparameters
        self.mu = nn.Parameter(torch.zeros(n_sounds,1).type(dtype))
        self.scale_hp = torch.Tensor(np.array(hyperpriors["scale"])).type(dtype)
        self.sigma_hp = torch.Tensor(np.array(hyperpriors["sigma"])).type(dtype)
        self._scale = nn.Parameter(torch.ones(n_sounds).type(dtype))
        self._sigma = nn.Parameter(torch.ones(n_sounds).type(dtype))

        self.dist_matrix = -1.0*torch.cdist(xpoints,xpoints)**2 #define this here because it never changes, don't need to recompute
        self.cov_eps = torch.full((xpoints.shape[1],),0.1).type(dtype)

        ## Values of Gaussian Process to infer
        self.vals = nn.Parameter(torch.zeros(n_sounds, xpoints.shape[1]).type(dtype))

    @property
    def scale(self):
        # Must always be positive
        return softplus(self._scale)

    @property
    def sigma(self):
        # Must always be positives
        return softplus(self._sigma)

    def cov(self):
        # Covariance matrix for squared exponential gaussian process
        sc = (self.scale**2)[:, np.newaxis, np.newaxis]
        si = (self.sigma**2)[:, np.newaxis, np.newaxis]
        covs = si * torch.exp( self.dist_matrix / sc )
        covs = covs + torch.diag(self.cov_eps**2)[np.newaxis, :, :]
        return covs

class NoiseElement(nn.Module):
    def __init__(self, timing, hyperpriors, audio_sr):
        super().__init__()

        ## Timing of element
        self.timing = timing #[onset, offset]
        self.duration = self.timing[1] - self.timing[0]

        ## Get x-values for GPs
        self.t = dr.get_element_gp_times(*self.timing, steps["t"])
        self.dt = len(self.t)
        self.x_t = torch.Tensor(np.reshape(self.t,(1,self.dt,1))).type(dtype)
        self.f = dr.get_element_gp_freqs(audio_sr, steps)
        self.df = len(self.f)
        self.x_f = torch.Tensor(np.reshape(self.f,(1,self.df,1))).type(dtype)

        ## Pre-define basic matrices needed for rendering
        # This will avoid re-computation of constants
        # Do not pre-define excitation, instead generate new noise on each iteration
        win_arr, self.sig_len = dr.make_overlapping_time_windows(self.dt+2, steps["t"], audio_sr)
        self.win_arr = torch.Tensor(win_arr[np.newaxis, :, :]).type(dtype)
        filterbank, bandwidthsHz, filt_freq_cutoffs = dr.make_overlapping_freq_windows(self.dt+2, steps, audio_sr)
        self.filterbank_numpy = filterbank
        self.filterbank = torch.Tensor(filterbank[np.newaxis,:,:]).type(dtype)
        self.bandwidthsHz = torch.Tensor(bandwidthsHz[np.newaxis, :, :]).type(dtype)

        ### Model latent variables: two 1-dimensional GPs, one for spectrum and one for amplitude
        self.spectrum = SquaredExponential(self.x_f, hyperpriors["spectrum"])
        self.amplitude = SquaredExponential(self.x_t, hyperpriors["amplitude"])

class NoiseScene(nn.Module):
    def __init__(self, observed_sound, audio_sr, timing, hyperpriors):
        super().__init__()
        """
        observed_sound: array of floats, soundwave
        audio_sr: integer, audio sampling rate
        timing: list of floats, [onset, offset]
        hyperpriors: dict of floats, for GP kernel
        """

        print("Defining dimensions...")
        self.n_sounds = 1
        self.steps = steps
        self.audio_sr = audio_sr
        self.rms_ref = rms_ref
        self.scene_len = observed_sound.shape[1]
        self.scene_duration = 1.0*self.scene_len/self.audio_sr

        print("Defining noise element...")
        self.element = NoiseElement(timing, hyperpriors, self.audio_sr)
        off_ramp = dr.hann_ramp(self.audio_sr)
        self.off_ramp = torch.Tensor(off_ramp[np.newaxis,:]).type(dtype)
        self.on_ramp = torch.flip(self.off_ramp, dims=(1,))

        print("Defining likelihood parameters...")
        self.eps_sigma = 10.0*torch.ones(self.n_sounds,1).type(dtype)

        print("Creating gammatonegram components...")
        self.nfft, self.nhop, self.nwin, gtm, gtg_freq_cutoffs = dg.gtg_settings()
        self.gtm = torch.Tensor(gtm[np.newaxis,:,:]).type(dtype);
        obs_sounds = torch.Tensor(observed_sound).type(dtype)
        self.window=torch.hann_window(self.nwin).type(dtype)
        self.obs = dg.gammatonegram(obs_sounds, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)

    ###Priors
    def score_filter(self, gps):
        loss = 0;
        for (i, gp) in enumerate([gps.spectrum, gps.amplitude]):
            loss -= MVN(gp.mu, covariance_matrix=gp.cov()).log_prob(gp.vals).sum()
            loss -= Gamma(*gp.scale_hp).log_prob(gp.scale).sum()
            loss -= Gamma(*gp.sigma_hp).log_prob(gp.sigma).sum()
        return loss

    ###Rendering
    def generate_sound(self, element):

        ## Get random excitation
        # Since we're not differentiating through the excitation, this can be generated in numpy (instead of torch)
        noise_excitation = dr.pink_excitation(element.sig_len/self.audio_sr, self.audio_sr)
        subbands = dr.generate_subbands(noise_excitation, element.filterbank_numpy)
        # Move subbands to torch
        subbands = torch.Tensor(subbands[np.newaxis,:,:]).type(dtype)

        ## Combine excitation and filter
        amplitude = torch.cat( (element.amplitude.vals[:,0][:, np.newaxis], element.amplitude.vals, element.amplitude.vals[:,-1][:,np.newaxis]), dim=1)
        F = element.spectrum.vals[:, np.newaxis, :] + amplitude[:, :, np.newaxis]
        n = dr.generate_noise(F, subbands, element.win_arr, element.filterbank, element.bandwidthsHz, element.duration, self.steps, self.audio_sr, self.rms_ref)

        ## Apply ramps to generated sound, for gradual onset and offsets
        n = n * torch.cat((self.on_ramp, torch.ones(self.n_sounds, n.shape[1]-self.on_ramp.shape[1]).type(dtype)), dim=1)
        n = n * torch.cat((torch.ones(self.n_sounds, n.shape[1]-self.off_ramp.shape[1]).type(dtype),self.off_ramp), dim=1)

        ## Put the element into the scene, according to its onset
        x = dr.register_element(n, element.timing, self.scene_len, self.scene_duration, self.audio_sr)

        return x

    def plot_step(self):
        return

    def forward(self):
        """
        Input
        =====
            obs (n_sounds, d_t, d_f)
        """

        ## Priors
        neg_log_prior = self.score_filter(self.element)

        ## Rendering and priors
        x = self.generate_sound(self.element)

        ## Likelihood
        gtg = dg.gammatonegram(x, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)
        neg_log_likelihood = -Normal(gtg.flatten(), self.eps_sigma).log_prob(self.obs.flatten()).sum()
        loss = neg_log_prior + neg_log_likelihood

        return loss, neg_log_prior, neg_log_likelihood, gtg, x

class HarmonicScene(nn.Module):
    def __init__(self, observed_sound, audio_sr, latents, regularization_weight=1e-4):
        super().__init__()

        """
        observed_sound: array of floats, soundwave
        audio_sr: integer, audio sampling rate
        latents: contains information to generate periodic excitation

        Note: this scene does not have priors!
        """

        print("Defining dimensions...")
        self.n_sounds = observed_sound.shape[0]
        self.steps = steps
        self.audio_sr = audio_sr
        self.rms_ref = rms_ref
        self.duration = 1.0*observed_sound.shape[1]/self.audio_sr

        print("\n\n\nobserved_sound:   ",observed_sound.shape)

        self.t = dr.get_element_gp_times(0.0, self.duration, self.steps["t"])
        self.dt = len(self.t)
        self.f = dr.get_element_gp_freqs(self.audio_sr, self.steps)
        self.df = len(self.f)

        print("Pre-compute rendering components...")
        win_arr, self.sig_len = dr.make_overlapping_time_windows(self.dt+2, self.steps["t"], self.audio_sr)
        self.win_arr = torch.Tensor(win_arr[np.newaxis, :, :]).type(dtype)
        filterbank, bandwidthsHz, filt_freq_cutoffs = dr.make_overlapping_freq_windows(self.dt+2, self.steps, self.audio_sr)
        self.filterbank = torch.Tensor(filterbank[np.newaxis,:,:]).type(dtype)
        self.bandwidthsHz = torch.Tensor(bandwidthsHz[np.newaxis, :, :]).type(dtype)
        print("Generate periodic excitation...")
        print("\n\n\n\n filt_freq_cutoffs:",filt_freq_cutoffs.shape)
        excitation_wave = dr.excitation_from_latents(observed_sound.shape[1], latents, self.audio_sr, self.steps, self.duration, self.win_arr.shape[1])
        x = excitation_wave
        f0 = latents[0,:,0]
        print('\n\nf0',f0)
        ex_freq = torch.zeros(len(f0),math.floor(audio_sr/(2*f0[0])))

        print('ex_freq',ex_freq.shape)
        for ii in range(0,len(f0)):
            ex_freq[ii,0:math.floor(audio_sr/(2*f0[ii]))] += f0[ii]*torch.arange(1,math.floor(audio_sr/(2*f0[ii]))+1,1)
        self.ex_freq = ex_freq

        filtbank_freq = (filt_freq_cutoffs[:,0] + filt_freq_cutoffs[:,1])/2
        self.filtbank_freq = filtbank_freq

        print('\n\n\nfiltbankfreq:',filtbank_freq)

        self.excitation = torch.Tensor(excitation_wave[np.newaxis, :]).type(dtype)


        print("Defining latent filter...")
        ##Filter parameter
        F_temp = torch.full((self.n_sounds,self.dt,self.df),1.0)

        self.F = nn.Parameter((torch.full((1,self.dt,self.df),0)- torch.arange(0,1,1.0/self.df).repeat(1,self.dt,1)).type(dtype)) # (n_sounds, d_f)

        self.spec_slope = nn.Parameter(torch.tensor(12).type(dtype))
        self.expo = torch.tensor(1).type(dtype)

        print(self.F)
        print("Defining mask for latent filter...")
        test1 = dr.edges(self.steps["t"],self.dt,2*self.steps["t"])


        F_mask = dr.make_mask(dr.edges(self.steps["t"],self.dt,2*self.steps["t"]), filt_freq_cutoffs, latents, self.df)
        self.F_mask = torch.Tensor((F_mask.T)[np.newaxis, :, :]).type(dtype)
        self.F_const = torch.full(self.F.shape,-60.0).type(dtype)

        print("Defining loss parameters...")
        self.regularization_weight = regularization_weight
        self.eps_sigma = 1.0*torch.ones(self.n_sounds,1).type(dtype) 

        print("Creating gammatonegram components...")
        ##Likelihood parameters
        twin=0.025; thop=0.010;
        self.nfft, self.nhop, self.nwin, gtm, gtg_freq_cutoffs = dg.gtg_settings(twin=twin,thop=thop)
        self.gtm = torch.Tensor(gtm[np.newaxis,:,:]).type(dtype);
        obs_sounds = torch.Tensor(observed_sound).type(dtype)
        self.window = torch.hann_window(self.nwin).type(dtype)
        self.obs = dg.gammatonegram(obs_sounds, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)


        print("Creating mask for likelihood computation...")
        self.obs1 = torch.mean(self.obs,2)

    def plot_step(self):
        return

    def forward(self):

        ###Rendering
        masked_F = dr.where(self.F_mask == 0, self.F, self.F)
        masked_F = torch.clamp(masked_F,min = -100,max = 100)
        F = torch.cat( (masked_F[:,0,:][:, np.newaxis, :], masked_F, masked_F[:,-1,:][:,np.newaxis,:]), dim=1)
        x = dr.generate_harmonic(self.excitation, F, self.win_arr, self.filterbank, self.bandwidthsHz, self.duration, self.steps, self.audio_sr, self.rms_ref,self.ex_freq,self.filtbank_freq,self.spec_slope,self.expo)
        x = x*1.00001;
        ###Regularization
        regularization_loss = (
            ((self.F[:, :-1, :] - self.F[:, 1:, :])**2).mean() +
            ((self.F[:, :, :-1] - self.F[:, :, 1:])**2).mean()
        ) * self.regularization_weight

        ###Likelihood
        gtg = dg.gammatonegram(x, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window)

        gtg1 = torch.mean(gtg,2)

        loss = (((gtg1.flatten()-self.obs1.flatten()))**2).mean()



        return loss, gtg, self.F, x
