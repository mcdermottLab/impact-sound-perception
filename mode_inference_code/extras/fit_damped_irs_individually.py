

#Arrays
import torch
import torch.nn as nn
from torch.optim import Adam,SGD,AdamW,RMSprop
import numpy as np
import numpy
#IO
from glob import glob
import os; import sys
import soundfile as sf
import fnmatch
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


homepath = "./"
sound_name = "glassIR"

soundpath = "./damp_rec_imps/"
num_sounds = 0
lim1 = 0
lim2 = 0



def load_observed_values(sound_name, expt_name):
    global num_sounds
    num_sounds = 0

 
    savepath = homepath + "{}_{}/".format(sound_name, expt_name)
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
        os.mkdir(savepath+"RT60/")
        os.mkdir(savepath+"Amp/")
        os.mkdir(savepath+"Alpha/")
        os.mkdir(savepath+"Gamma/")
        os.mkdir(savepath+"freq/")

    # Listing all wave files in the soundpath directory and its subdirectories
    dirs = []; names = []
    jjj = 0
    for dirpath, dirnames, filenames in os.walk(soundpath):
        for filename in fnmatch.filter(filenames, '*.wav'):
                dirs.append(dirpath)
                names.append(filename)
                jjj = jjj + 1
    print("jjj",jjj)
    print("names",names[0])        


    onlyfiles = [f for f in os.listdir(soundpath) if os.path.isfile(os.path.join(soundpath, f))]
    numpy.savetxt(savepath+ "dirnames.csv",dirs, delimiter = ',', fmt="%s")
    numpy.savetxt(savepath+ "filenames.csv",names, delimiter = ',', fmt="%s")
    sound_dur = 1.0
    observed_audio = np.zeros([len(dirs),int(sound_dur*44100)]); ind1 = 0
    for ind in range(0,len(dirs)):
        observed_audio1, audio_sr = sf.read(dirs[ind] + "/" + names[ind])
        observed_audio1 = clean_audio(observed_audio1)
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

    num_sounds += observed_audio.shape[0]
    print("num_sounds",num_sounds)

    #load the variable in file ind_40.csv and save it in indd2

    

    return observed_audio, audio_sr, timing, hyperpriors, savepath,latents

def clean_audio(observed_audio):
    sq_audio = observed_audio**2
    sq_audio = sq_audio-np.mean(sq_audio)
    sq_audio = sq_audio/np.max(abs(sq_audio))
    non_zer = np.nonzero(abs(sq_audio)>1e-6)
    max_val = np.argmax(abs(sq_audio),axis=0)

    print(non_zer[0].shape)
    
    fin_audio = observed_audio[max_val:]
    fin_audio = fin_audio-np.mean(fin_audio)
    fin_audio = fin_audio/np.max(abs(fin_audio))
    
    return fin_audio

def initialize_model(sound_name, expt_name,snd):

    observed_audio, audio_sr, timing, hyperpriors, savepath,latents = load_observed_values(sound_name, expt_name)

    audio_use = observed_audio[snd][np.newaxis,:]
    scene = ImpactSound(audio_use, audio_sr,0.000001)
    scene.train()
    if torch.cuda.is_available(): 
        scene.cuda()
    for (k, v) in scene.named_parameters():
        print(k, v)

    lr_start = 10e-5
    optimizer = RMSprop(scene.parameters(),lr=lr_start)
    print("printing params")
    for p in scene.named_parameters():
        print(p)
    print("printing params done")

    return scene, optimizer, savepath

def train(sound_name, expt_name, total_steps=500000):

    global lim1, lim2

    print("lim1",lim1)
    print("lim2",lim2)
    indd2 = np.loadtxt("./inds_30_2.csv", delimiter = ',')

    for snd1 in range(lim1,lim2):
        snd = int(snd1)
        scene, optimizer, savepath = initialize_model(sound_name, expt_name,snd)
        
        losses = []; gps = []; hps = []; priors = []; likelihoods = []; RT60_all = []
        for step_idx in range(1,total_steps):

            optimizer.zero_grad()
            loss, masked_gtg, x = scene.forward(step_idx)
            if loss.isnan():
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/RT60.csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Amp.csv", scene.Amp.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/freq.csv", scene.mode_freq.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/alpha.csv", scene.alpha.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/gamma.csv", scene.gamma.detach().cpu().numpy(), delimiter = ',')
                break

            


            

            loss.backward()

            nn.utils.clip_grad_norm_(scene.parameters(), max_norm=50.0)
            optimizer.step()
            losses.append(loss.item())
            savepath = "./"+sound_name+"_"+expt_name +"/"+str(snd)
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
                os.mkdir(savepath+"/RT60")
                os.mkdir(savepath+"/Amp")
                os.mkdir(savepath+"/Alpha")
                os.mkdir(savepath+"/Gamma")
                os.mkdir(savepath+"/freq")


            if step_idx == 1:
                obs_audio = scene.observed_sound.detach().cpu()
                torchaudio.save(savepath+ "/observed_" + str(step_idx) +'.wav',0.5*obs_audio/obs_audio.max(),44100)
            

            if step_idx % 100== 0:
                # Make a directory for the current sound
                
                ## Print current loss
                print("Step {:05d} -- loss {:.6f}".format(step_idx, loss))
                # save loss as a figure
                plt.figure()
                plt.plot(losses)
                plt.savefig("./"+sound_name+"_"+expt_name +"/"+str(snd)+ "/loss.png")
                plt.close()
            

            if step_idx % 1000 == 0:
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/RT60/RT60_" + sound_name + str(step_idx) + ".csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Amp/Amp_" + sound_name + str(step_idx) + ".csv", scene.Amp.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/freq/freq_" + sound_name + str(step_idx) + ".csv", scene.mode_freq.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Alpha/alpha_" + sound_name + str(step_idx) + ".csv", scene.alpha.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Gamma/gamma_" + sound_name + str(step_idx) + ".csv", scene.gamma.detach().cpu().numpy(), delimiter = ',')

            
            if step_idx % 500 == 1:
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/RT60.csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Amp.csv", scene.Amp.squeeze().detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/freq.csv", scene.mode_freq.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/alpha.csv", scene.alpha.detach().cpu().numpy(), delimiter = ',')
                numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/gamma.csv", scene.gamma.detach().cpu().numpy(), delimiter = ',')
                print("")
                if loss<3:
                    break
                
                
                for index in range(0,x.shape[0]):
                    savepath = "./"+sound_name+"_"+expt_name+"/"+str(snd)+ "/"
                    if not os.path.isdir(savepath):
                        os.mkdir(savepath)
                    x1 = x.cpu()
                    
                    torchaudio.save(savepath+ "output_" + str(step_idx) +'.wav',0.5*x1[index][np.newaxis,:]/x1[index].max(),44100)
                    #save observed sound to a wav file
                    

            if step_idx%20000==19999:
                if (losses[-8999] - loss) < 0.000001*loss:
                    numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/RT60.csv", scene.RT60.squeeze().detach().cpu().numpy(), delimiter = ',')
                    numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/Amp.csv", scene.Amp.squeeze().detach().cpu().numpy(), delimiter = ',')
                    numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/freq.csv", scene.mode_freq.detach().cpu().numpy(), delimiter = ',')
                    numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/alpha.csv", scene.alpha.detach().cpu().numpy(), delimiter = ',')
                    numpy.savetxt("./"+sound_name+"_"+expt_name+"/"+str(snd)+"/gamma.csv", scene.gamma.detach().cpu().numpy(), delimiter = ',')
                    break



                

    print("Complete!")


if __name__ == '__main__':
    sound_name   = sys.argv[1]
    expt_name= sys.argv[2]
    load_observed_values(sound_name, expt_name)
    lim1 = (int(sys.argv[3]) -1)
    lim2 = lim1 + 1
    
    train(sound_name, expt_name,100000)
