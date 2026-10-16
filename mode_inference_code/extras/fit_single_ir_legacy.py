

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

vis = visdom.Visdom(env="Audio debugging")

homepath = "./"
soundpath = "./"



def load_observed_values(sound_name, expt_name):

    observed_audio, audio_sr = sf.read(soundpath + sound_name + ".wav")

    observed_audio = observed_audio[np.newaxis, :]

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
    scene = ImpactSound(observed_audio, audio_sr, latents,0.000001)
    scene.train()
    for (k, v) in scene.named_parameters():
        print(k, v)

    lr_start = 8e-6
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
        x = x.flatten()
        x= x[np.newaxis,:]
        numpy.savetxt("./"+sound_name+"_"+expt_name+"/freq_" + sound_name + ".csv", scene.mode_freq.squeeze().detach().numpy(), delimiter = ',')

        loss.backward();


        optimizer.step()
        losses.append(loss.item())


        if step_idx % 1000== 0:
            ## Print current loss
            print("Step {:05d} -- loss {:.6f}".format(step_idx, loss), flush=True)
        
        RT60_np = scene.RT60.squeeze().detach().numpy()
        RT60_all.append(RT60_np[0])

        if step_idx % 1000 == 0:
            numpy.savetxt("./"+sound_name+"_"+expt_name+"/RT60/RT60_" + sound_name + str(step_idx) + ".csv", scene.RT60.squeeze().detach().numpy(), delimiter = ',')
            print("")

        if step_idx % 1000 == 0:
            torch.save({
                'step_idx': step_idx,
                'scene_state_dict': scene.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, savepath + "ckpt-{:08d}.pth".format(step_idx))
            torchaudio.save("./"+sound_name+"_"+expt_name+"/output_" + str(step_idx) +'.wav',0.5*x/x.max(),44100)
            vis.audio(audiofile="./"+sound_name+"_"+expt_name+"/output_" + str(step_idx) +'.wav', win="Latest Audio")
            vis.audio(audiofile="./"+sound_name+'.wav', win="Original Audio")



        if step_idx % 100 == 0:
            vis.line(losses, np.arange(len(losses)), opts={"title": "Losses"}, win="Losses") 
            
            vis.line(RT60_all, np.arange(len(RT60_all)), opts={"title": "RT60_all"}, win="RT60_all") 

    numpy.savetxt("RT60_" + sound_name + ".csv", scene.RT60.squeeze().detach().numpy(), delimiter = ',')
    numpy.savetxt("freq_" + sound_name + ".csv", scene.mode_freq.squeeze().detach().numpy(), delimiter = ',')
    print("Complete!")


if __name__ == '__main__':
    sound_name   = sys.argv[1]
    expt_name= sys.argv[2]
    train(sound_name, expt_name)
