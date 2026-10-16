"""
PyTorch routine for inferring latent variable in sound scene
Maddie Cusimano, mcusi@mit.edu November 2020
"""

#Arrays
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
import numpy as np
import numpy
import math
#IO
from glob import glob
import os; import sys
import soundfile as sf
import json
import torchaudio
#Custom Code
from impact_model import HarmonicScene
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sound_renderer as dr

homepath = "./"
soundpath = "./trombonef/"

def load_observed_values(sound_name, expt_name):
    sound_num = 3
    shift = 9
    observed_audio = torch.zeros(sound_num,math.floor(1.5*44100))
    print('sound:   ',len(sound_name))
    for inddd in range(0,sound_num):
        observed_audio1, audio_sr = sf.read(soundpath + "note"+str(inddd+shift+1) + ".wav")
        observed_audio[inddd] = observed_audio[inddd] + observed_audio1[0:math.floor(1.5*44100)]

    hyperpriors = {"spectrum":{"scale": [8.952, 1/0.4384], "sigma": [63.6599, 1/0.2382]}, "amplitude":{"scale": [1.2915, 1/0.6833], "sigma": [0.0926, 1/10.6342]}}
    timing = [0.1, 0.5]
    natfreq  = 60
    latents = torch.zeros((1,sound_num,3))
    for ind1 in range(0,sound_num):
        latents[0,ind1,0] = 63 * 2**((ind1+shift)/12)
        latents[0,ind1,1] = 0
        latents[0,ind1,2] = 2

    savepath = homepath + "{}_{}/".format(sound_name, expt_name)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    return observed_audio, audio_sr, timing, hyperpriors, savepath,latents

def initialize_model(sound_name, expt_name):

    observed_audio, audio_sr, timing, hyperpriors, savepath,latents = load_observed_values(sound_name, expt_name)
    scene = HarmonicScene(observed_audio, audio_sr, latents, 0.01)
    scene.train()
    optimizer = Adam(scene.parameters(),lr=0.03)

    print("printing params")
    for p in scene.named_parameters():
        print(p)
    print("printing params done")

    return scene, optimizer, savepath

def train(sound_name, expt_name, total_steps=1000):

    scene, optimizer, savepath = initialize_model(sound_name, expt_name)
    losses = []; gps = []; hps = []; priors = []; likelihoods = []
    for step_idx in range(1,total_steps):

        optimizer.zero_grad()
        loss, masked_gtg, masked_F, x = scene.forward()

        loss.backward();


        if torch.isnan(scene.F.grad.min()):
            numpy.savetxt("Final_F_tf_3_tot2.csv", scene.F[0,0,:].detach().numpy(), delimiter = ',')
            numpy.savetxt("Final_Freq_tf_3_tot2.csv", scene.filtbank_freq, delimiter = ',')
            exit(0)

        optimizer.step()
        losses.append(loss.item())


        if step_idx % 5 == 0:
            ## Print current loss
            print("Step {:05d} -- loss {:.6f}".format(step_idx, loss), flush=True)
            print(scene.F[0,0,:])
            print('\n\n\nsigma:   ',scene.spec_slope)

        if step_idx % 100 == 0:
            print("")

        if step_idx % 50 == 0:
            torch.save({
                'step_idx': step_idx,
                'scene_state_dict': scene.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, savepath + "ckpt-{:08d}.pth".format(step_idx))
            numpy.savetxt("Final_F_tf_3_tot2.csv", scene.F[0,0,:].detach().numpy(), delimiter = ',')
            numpy.savetxt("Final_Freq_tf_3_tot2.csv", scene.filtbank_freq, delimiter = ',')


    print("Complete!")


if __name__ == '__main__':

    sound_name = ["","","",""]
    sound_name[0]   = sys.argv[1]
    sound_name[1] = sys.argv[2]
    sound_name[2] = sys.argv[3]
    sound_name[3] = sys.argv[4]
    expt_name= sys.argv[5]
    train(sound_name, expt_name)
