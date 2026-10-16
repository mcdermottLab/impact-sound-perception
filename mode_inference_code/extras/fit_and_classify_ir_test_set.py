

#Arrays
import torch
import torch.nn as nn
from torch.optim import Adam,SGD,AdamW
import numpy as np
import numpy
#IO
from glob import glob
import os; import sys
import soundfile as sf
import json
import torchaudio
#Custom Code
from impact_model import ImpactSound
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sound_renderer as dr
from numpy import genfromtxt
from torch.distributions.multivariate_normal import MultivariateNormal as MVN



homepath = "./"
soundpath = "./test_set_v2/"

onlyfiles = [f for f in os.listdir(soundpath) if os.path.isfile(os.path.join(soundpath, f))]
sounds = ["" for x in range(5)]; ind1 = 0

for ind in range(0,len(onlyfiles)):
        if onlyfiles[ind][-4:]==".wav":
            sounds[ind1] = onlyfiles[ind]
            ind1 = ind1+1


def load_observed_values(sound_name, expt_name):

    
    sound_dur = 1
    observed_audio = np.zeros([len(onlyfiles),sound_dur*44100]); ind1 = 0
    for ind in range(0,len(onlyfiles)):
        if onlyfiles[ind][-4:]==".wav":
            observed_audio1, audio_sr = sf.read(soundpath + onlyfiles[ind])
            if len(observed_audio1)> sound_dur*44100:
                observed_audio[ind1] = observed_audio1[:sound_dur*44100]
            else:
                observed_audio[ind1,:len(observed_audio1)] = observed_audio1
            ind1 = ind1 +1

    

    audio_sr =44100


    hyperpriors = {"spectrum":{"scale": [8.952, 1/0.4384], "sigma": [63.6599, 1/0.2382]}, "amplitude":{"scale": [1.2915, 1/0.6833], "sigma": [0.0926, 1/10.6342]}}
    timing = [0.1, 0.5]
    natfreq  = 60
    latents = torch.zeros((1,2,3))
    ind1 =1
    
    savepath = homepath + "{}_{}/".format(sound_name, expt_name)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath+"RT60/")
        os.mkdir(savepath+"Amp/")

    return observed_audio, audio_sr, timing, hyperpriors, savepath,latents

def clean_audio(observed_audio):
    sq_audio = observed_audio**2
    
    non_zer = np.nonzero(sq_audio>1e-9)
    fin_audio = observed_audio[non_zer[0][0]:]
    fin_audio = np.append(fin_audio,np.zeros(100100-len(fin_audio)))
    print(non_zer[0])
    
    return fin_audio

def initialize_model(sound_name, expt_name):

    observed_audio, audio_sr, timing, hyperpriors, savepath,latents = load_observed_values(sound_name, expt_name)

    scene = ImpactSound(observed_audio, audio_sr,0.000001)
    scene.train()
    if torch.cuda.is_available(): 
        scene.cuda()
    for (k, v) in scene.named_parameters():
        print(k, v)

    lr_start = 5e-5
    optimizer = AdamW(scene.parameters(),lr=lr_start)
    print("printing params")
    for p in scene.named_parameters():
        print(p)
    print("printing params done")

    return scene, optimizer, savepath

def train(sound_name, expt_name, total_steps=100000):

    scene, optimizer, savepath = initialize_model(sound_name, expt_name)
    losses = []; gps = []; hps = []; priors = []; likelihoods = []; RT60_all = []
    new_lr = 5e-5
    for step_idx in range(1,total_steps):

        optimizer.zero_grad()
        loss, masked_gtg, x = scene.forward(step_idx)
        if step_idx == 1:
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/freq_" + sound_name + ".csv", scene.mode_freq.detach().cpu().numpy(), delimiter = ',')

        loss.backward();
        
        optimizer.step()
        losses.append(loss.item())

        if torch.isnan(loss):
            break



        if step_idx % 2000== 0:
            ## Print current loss
            print("Step {:05d} -- loss {:.6f}".format(step_idx, loss), flush=True)
        
        RT60_np = scene.RT60[0].detach().cpu().numpy()
        RT60_all.append(RT60_np)

        if step_idx % 5000 == 0:
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/RT60/RT60_" + sound_name + str(step_idx) + ".csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/Amp/Amp_" + sound_name + str(step_idx) + ".csv", scene.Amp.squeeze().detach().cpu().numpy(), delimiter = ',')
            print("")

        if step_idx % 100 == 0:
            torch.save({
                'step_idx': step_idx,
                'scene_state_dict': scene.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, savepath + "ckpt-{:08d}.pth".format(step_idx))
            
            
            for index in range(0,x.shape[0]):
                savepath = "./"+sound_name+"_"+expt_name+"/"+str(index+1) + "/"
                if not os.path.isdir(savepath):
                    os.mkdir(savepath)



            

        modes = torch.cat((scene.mode_freq[:,np.newaxis,:], -20*torch.log10(torch.abs(scene.Amp[:,np.newaxis,:])+1e-4), scene.RT60[:,np.newaxis,:]),1)

    numpy.savetxt("RT60_" + sound_name + ".csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
    numpy.savetxt("freq_" + sound_name + ".csv", scene.mode_freq.squeeze().detach().cpu().numpy(), delimiter = ',')
    print("Complete!")

    return modes


def infer_mat(optimize_modes):
    #load csvs with material properties
    #calculate the mean and covariance of the modal properties
    #calculate the probabilities of the current modes from the multivariate normal distribution

    onlyfiles = [f for f in os.listdir(soundpath) if os.path.isfile(os.path.join(soundpath, f))]
    batch = optimize_modes.shape[0]
    n_modes_per_object = 10
    n_objects = 4
    class_log_probs = torch.zeros([4, n_modes_per_object]).to(optimize_modes.device)
    load_dir = "./test_num2/"
    for b in range(0,batch):
        optimized_modes = optimize_modes[b]
        sound_name1 = sounds[b]    
       
        for count,material in enumerate(["Glass", "Metal", "Plastic", "Wood"]):
            mat_freq = genfromtxt(load_dir+"freq_"+material+".csv",delimiter=",")[np.newaxis,:]
            mat_Amp = genfromtxt(load_dir+"Amp_"+material+".csv",delimiter=",")
            mat_RT60 = 1000*genfromtxt(load_dir+"RT60_"+material+".csv",delimiter=",")[np.newaxis,:]


            mat_Amp = -20*np.log10(np.abs(mat_Amp)+1e-4)
            mat_Amp = mat_Amp[np.newaxis,:]
            mat_modes = np.concatenate((mat_freq, mat_Amp, mat_RT60))


            #Consider the modes independently
            mat_modes = np.transpose(mat_modes.reshape(3,n_modes_per_object*n_objects))
            mu = np.mean(mat_modes,0)#Mean: f, a, RT60
            cov = np.cov(np.transpose(mat_modes)) + 0.001*np.identity(3)
            material_dist = MVN(torch.Tensor(mu).to(optimized_modes.device), torch.Tensor(cov).to(optimized_modes.device))

            for ind in range(10):
                log_p = material_dist.log_prob(optimized_modes[:,ind])
                class_log_probs[count,ind] =  log_p
        sum_log_probs = torch.sum(class_log_probs,1)
        normalized_class_probabilities = torch.exp(sum_log_probs-torch.logsumexp(sum_log_probs,0))
        print("probabilities:  ",normalized_class_probabilities)
        numpy.savetxt("./"+sound_name+"_"+expt_name+"/probs_" + sound_name1 + ".csv", normalized_class_probabilities.squeeze().detach().cpu().numpy(), delimiter = ',')
        





if __name__ == '__main__':
    sound_name   = sys.argv[1]
    expt_name= sys.argv[2]
    opt_modes = train(sound_name, expt_name)
    infer_mat(opt_modes)

