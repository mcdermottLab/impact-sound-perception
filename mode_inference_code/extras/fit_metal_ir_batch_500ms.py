

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
import visdom
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

vis = visdom.Visdom(env="Audio debugging")

homepath = "./"
soundpath = "./Metal_learn/"



def load_observed_values(sound_name, expt_name):

    onlyfiles = [f for f in os.listdir(soundpath) if os.path.isfile(os.path.join(soundpath, f))]
    sound_dur = 0.5
    observed_audio = np.zeros([len(onlyfiles)-1,int(sound_dur*44100)]); ind1 = 0
    for ind in range(0,len(onlyfiles)):
        if onlyfiles[ind][-3:]=="wav":
            observed_audio1, audio_sr = sf.read(soundpath + onlyfiles[ind])
            if len(observed_audio1)> sound_dur*44100:
                observed_audio[ind1] = observed_audio1[:int(sound_dur*44100)]
            else:
                observed_audio[ind1,:len(observed_audio1)] = observed_audio1
            ind1 = ind1 +1

    

    


    hyperpriors = {"spectrum":{"scale": [8.952, 1/0.4384], "sigma": [63.6599, 1/0.2382]}, "amplitude":{"scale": [1.2915, 1/0.6833], "sigma": [0.0926, 1/10.6342]}}
    timing = [0.1, 0.5]
    natfreq  = 60
    latents = torch.zeros((1,2,3))
    ind1 =1
    latents[0,0,0] = 58. * 2**((ind1)/12)
    latents[0,0,1] = 0.01
    latents[0,0,2] = 7

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
    for (k, v) in scene.named_parameters():
        print(k, v)

    lr_start = 8e-5
    optimizer = AdamW(scene.parameters(),lr=lr_start)
    print("printing params")
    for p in scene.named_parameters():
        print(p)
    print("printing params done")

    return scene, optimizer, savepath

def train(sound_name, expt_name, total_steps=500000):

    scene, optimizer, savepath = initialize_model(sound_name, expt_name)
    losses = []; gps = []; hps = []; priors = []; likelihoods = []; RT60_all = []
    new_lr = 5e-5
    for step_idx in range(1,total_steps):

        optimizer.zero_grad()
        loss, masked_gtg, x = scene.forward(step_idx)
        if step_idx == 1:
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/freq_" + sound_name + ".csv", scene.mode_freq.detach().numpy(), delimiter = ',')

        loss.backward();


        optimizer.step()
        losses.append(loss.item())


        if step_idx % 2== 0:
            ## Print current loss
            print("Step {:05d} -- loss {:.6f}".format(step_idx, loss), flush=True)
        
        RT60_np = scene.RT60[0].detach().numpy()
        RT60_all.append(RT60_np)

        if step_idx % 200 == 0:
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/RT60/RT60_" + sound_name + str(step_idx) + ".csv", scene.RT60.squeeze().detach().numpy(), delimiter = ',')
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/Amp/Amp_" + sound_name + str(step_idx) + ".csv", scene.Amp.squeeze().detach().numpy(), delimiter = ',')
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
                torchaudio.save(savepath+ "output_" + str(step_idx) +'.wav',0.5*x[index][np.newaxis,:]/x[index].max(),44100)
                vis.audio(audiofile=savepath+ "output_" + str(step_idx) +'.wav', win="Audio"+str(index+1))



        if step_idx % 2 == 0:
            vis.line(losses, np.arange(len(losses)), opts={"title": "Losses"}, win="Losses") 
            

    numpy.savetxt("RT60_" + sound_name + ".csv", scene.RT60.squeeze().detach().numpy(), delimiter = ',')
    numpy.savetxt("freq_" + sound_name + ".csv", scene.mode_freq.squeeze().detach().numpy(), delimiter = ',')
    print("Complete!")


if __name__ == '__main__':
    sound_name   = sys.argv[1]
    expt_name= sys.argv[2]
    train(sound_name, expt_name)
