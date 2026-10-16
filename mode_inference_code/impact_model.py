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
        num_modes = 10
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
        #reshape kernel
        kernel_n = kernel.unsqueeze(0).unsqueeze(0).repeat(1,self.n_sounds,1).type(dtype)
        #size of kernel_n = [1, n_sounds, kernel_size]
        #size of self.obs_rfft = [n_sounds, n_freq, n_time]
        self.obs_rfft = F.conv1d(self.obs_rfft, kernel_n)
        filter_params = {
            'sr': 44100,
            'nfft': self.nfft,
            'n_channels': 20,
            'low_lim': 20,
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
        
        return loss, gtg, x




