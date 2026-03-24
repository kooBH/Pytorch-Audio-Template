import os,glob,sys
import librosa as rs
import soundfile as sf
import numpy as np
import random
from tqdm.auto import tqdm

import data.Mixer as m
from utils.hparams import HParam

def match_length(x,len_data,idx_start=None) : 
    if len(x) > len_data : 
        left = len(x) - len_data
        if idx_start is None :
            idx_start = np.random.randint(left)
        x = x[idx_start:idx_start+len_data]
    elif len(x) < len_data : 
        shortage = len_data - len(x) 
        x = np.pad(x,(0,shortage))
    return x, idx_start

if __name__ == "__main__":
    hp = HParam("config/data/VD.yaml")

    list_clean = glob.glob(os.path.join(hp.root_clean,"**","*.wav"),recursive=True)
    list_noise = glob.glob(os.path.join(hp.root_noise,"**","*.wav"),recursive=True)
    list_RIR = glob.glob(os.path.join(hp.RIR,"**","*.wav"),recursive=True)

    save_dir = os.path.join(hp.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"noisy"), exist_ok=True)
    os.makedirs(os.path.join(save_dir,"clean"), exist_ok=True)

    print(f"Clean : {len(list_clean)} | Noise : {len(list_noise)}")

    for i,clean_path in enumerate(tqdm(list_clean)) :
        clean, sr = rs.load(clean_path, sr=hp.sr)
        noise_path = random.choice(list_noise)
        noise, sr = rs.load(noise_path, sr=hp.sr)

        clean , _ = match_length(clean, hp.len_data)
        noise , _ = match_length(noise, hp.len_data)

        rir = None
        if hp.Reverb.use : 
            if random.random() < hp.Reverb.prob:
                rir_path = random.choice(list_RIR)
                rir, sr = rs.load(rir_path, sr=hp.sr)

        noisy, clean, noise = m.mix(
            clean,noise,
            rir = rir,
            target_dB_FS = hp.Scale.target_dB_FS,
            target_dB_FS_floating_value = hp.Scale.target_dB_FS_floating_value,
            scale_method = hp.Scale.method,
            range_SNR=hp.SNR,
            deverb_clean=hp.Reverb.deverb_clean,
            clean_rir_len=hp.Reverb.clean_rir_len,
            use_spec_augmentation = False,
        )

        sf.write(os.path.join(save_dir,"noisy",f"{i}.wav"), noisy, sr)
        sf.write(os.path.join(save_dir,"clean",f"{i}.wav"), clean, sr)




