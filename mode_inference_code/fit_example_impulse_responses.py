#Arrays
import argparse
from pathlib import Path
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

from torch.distributions.multivariate_normal import MultivariateNormal as MVN


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
homepath = str(REPOSITORY_ROOT / "outputs" / "modal_fits") + os.sep
sound_name = "glassIR"

soundpath = str(REPOSITORY_ROOT / "data" / "example_IRs")
input_files = None
num_sounds = 0
lim1 = 0
lim2 = 0




def load_observed_values(sound_name, expt_name):
    global num_sounds
    num_sounds = 0

    savepath = homepath + "{}_{}/".format(sound_name, expt_name)
    Path(savepath).mkdir(parents=True, exist_ok=True)

    dirs = []; names = []
    jjj = 0
    paths = input_files
    if paths is None:
        paths = sorted(Path(soundpath).rglob("h.wav"))
    for path in paths:
        dirs.append(str(path.parent))
        names.append(path.name)
        jjj += 1
    if not names:
        raise ValueError("No input WAV files found. Use --input with a WAV file or a directory containing h.wav files.")
    print("jjj",jjj)
    print("names",names[0])        

    numpy.savetxt(savepath+ "dirnames.csv",dirs, delimiter = ',', fmt="%s")
    numpy.savetxt(savepath+ "filenames.csv",names, delimiter = ',', fmt="%s")
    sound_dur = 0.5
    observed_audio = np.zeros([len(dirs),int(sound_dur*44100)]); ind1 = 0
    for ind in range(0,len(dirs)):
        observed_audio1, audio_sr = sf.read(dirs[ind] + "/" + names[ind])
        if audio_sr != 44100 or observed_audio1.ndim != 1:
            raise ValueError(f"Expected a mono 44100-Hz WAV: {dirs[ind]}/{names[ind]}")
        observed_audio1 = clean_audio(observed_audio1)
        if len(observed_audio1)> sound_dur*44100:
            observed_audio[ind1] = observed_audio1[:int(sound_dur*44100)]
        else:
            observed_audio[ind1,:len(observed_audio1)] = observed_audio1
        ind1 = ind1 +1

    num_sounds += observed_audio.shape[0]
    print("num_sounds",num_sounds)

    return observed_audio, audio_sr, savepath

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

def save_parameters(scene, output_directory, run_name, step_idx=None):
    """Save current modal parameters, creating checkpoint folders when needed."""
    output_directory = Path(output_directory)
    for parameter, directory, prefix in (
        ("RT60", "RT60", "RT60"), ("Amp", "Amp", "Amp"),
        ("mode_freq", "freq", "freq"), ("alpha", "Alpha", "alpha"),
        ("gamma", "Gamma", "gamma"),
    ):
        if step_idx is None:
            destination = output_directory / f"{prefix}.csv"
        else:
            destination = output_directory / directory / f"{prefix}_{run_name}{step_idx}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = getattr(scene, parameter).detach().cpu().numpy()
        if parameter in ("RT60", "Amp"):
            values = values.squeeze()
        numpy.savetxt(destination, values, delimiter=",")

def initialize_model(sound_name, expt_name,snd):

    observed_audio, audio_sr, savepath,= load_observed_values(sound_name, expt_name)

    audio_use = observed_audio[snd][np.newaxis,:]
    scene = ImpactSound(audio_use, audio_sr,0.000001)
    scene.train()
    if torch.cuda.is_available(): 
        scene.cuda()
    for (k, v) in scene.named_parameters():
        print(k, v)

    lr_start = 10e-5
    optimizer = Adam(scene.parameters(),lr=lr_start)
    print("printing params")
    for p in scene.named_parameters():
        print(p)
    print("printing params done")

    return scene, optimizer, savepath

def train(sound_name, expt_name, total_steps=500000):

    global lim1, lim2

    print("lim1",lim1)
    print("lim2",lim2)

    for snd1 in range(lim1,lim2):
        snd = int(snd1)
        scene, optimizer, savepath = initialize_model(sound_name, expt_name,snd)
        sound_directory = Path(savepath) / str(snd)
        for subdirectory in ("", "RT60", "Amp", "Alpha", "Gamma", "freq"):
            (sound_directory / subdirectory).mkdir(parents=True, exist_ok=True)
        print(f"Output for input {snd}: {sound_directory.resolve()}")
        
        losses = []; gps = []; hps = []; priors = []; likelihoods = []; RT60_all = []

        for step_idx in range(1,total_steps):

            optimizer.zero_grad()
            loss, masked_gtg, x = scene.forward(step_idx)
            if loss.isnan():
                save_parameters(scene, sound_directory, sound_name)
                break
            

            loss.backward()

            nn.utils.clip_grad_norm_(scene.parameters(), max_norm=50.0)
            optimizer.step()
            losses.append(loss.item())
            savepath = str(sound_directory)


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
                plt.savefig(str(sound_directory / "loss.png"))
                plt.close()
                

            if step_idx % 1000 == 0:
                save_parameters(scene, sound_directory, sound_name, step_idx)

            
            if step_idx % 500 == 1:
                save_parameters(scene, sound_directory, sound_name)
                print("")
                if loss<3:
                    break
                
                
                for index in range(0,x.shape[0]):
                    savepath = str(sound_directory) + os.sep
                    if not os.path.isdir(savepath):
                        os.mkdir(savepath)
                    # Detach tensors before saving
                    x1 = x.detach().cpu()
                    
                    torchaudio.save(savepath+ "output_" + str(step_idx) +'.wav',0.5*x1[index][np.newaxis,:]/x1[index].max(),44100)
                    #save observed sound to a wav file
                    

            if step_idx%20000==19999:
                if (losses[-8999] - loss) < 0.000001*loss:
                    save_parameters(scene, sound_directory, sound_name)
                    break
            
    print("Complete!")


def main(argv=None):
    """Fit the included example IRs or a specified mono 44100-Hz WAV."""
    global homepath, soundpath, input_files, lim1, lim2
    parser = argparse.ArgumentParser(description=__doc__ or main.__doc__)
    parser.add_argument("run_name", nargs="?", default="example_IRs", help="Output label, not an input path (default: example_IRs).")
    parser.add_argument("experiment_name", nargs="?", default="demo", help="Second output label (default: demo).")
    parser.add_argument("--input", type=Path, default=REPOSITORY_ROOT / "data" / "example_IRs", help="A mono 44100-Hz WAV, or a directory searched recursively for h.wav files.")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "modal_fits", help="Parent directory for <run_name>_<experiment_name> outputs.")
    parser.add_argument("--num-sounds", type=int, default=2, help="Maximum number of inputs to fit (default: 2).")
    parser.add_argument("--start-index", type=int, default=0, help="First input index to fit (default: 0).")
    parser.add_argument("--steps", type=int, default=100000, help="Optimization loop limit; early stopping is retained (default: 100000).")
    args = parser.parse_args(argv)
    for label in (args.run_name, args.experiment_name):
        if label in (".", "..") or not label or any(not (c.isalnum() or c in "_.-") for c in label):
            parser.error("Run and experiment names must be simple labels. Supply input paths with --input.")
    if args.num_sounds < 1 or args.start_index < 0 or args.steps < 2:
        parser.error("--num-sounds must be positive, --start-index nonnegative, and --steps at least 2.")
    source = args.input.expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".wav":
        input_files = [source]
    elif source.is_dir():
        input_files = sorted(source.rglob("h.wav"))
    else:
        parser.error(f"Input must be an existing WAV or directory: {source}")
    if not input_files or args.start_index >= len(input_files):
        parser.error("No input WAVs available at the requested start index.")
    soundpath = str(source if source.is_dir() else source.parent)
    homepath = str(args.output_dir.expanduser().resolve()) + os.sep
    lim1 = args.start_index
    lim2 = min(len(input_files), lim1 + args.num_sounds)
    print(f"Fitting {lim2-lim1} input(s); output directory: {Path(homepath) / (args.run_name + '_' + args.experiment_name)}")
    for index in range(lim1, lim2):
        print(f"  {index}: {input_files[index]}")
    train(args.run_name, args.experiment_name, args.steps)


if __name__ == '__main__':
    main()
