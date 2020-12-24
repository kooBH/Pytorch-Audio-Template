import torch
import argparse
import numpy as np
import torchaudio
import os
import sys
import glob
import math
from fairseq import utils
from model.DCUnet_jsdr_demand import *
import librosa
from tensorboardX import SummaryWriter
from datasets.testDataset import *
import scipy.io.wavfile

from tqdm import tqdm

from utils.hparams import HParam

def tensor2audio(audio,window,length):
    window = window
    length = length
    audio = audio
    audio = audio.numpy().squeeze()
    return audio
    
def complex_demand_audio(complex_ri,window,length,fs):
    window = window
    length = length
    complex_ri = complex_ri
    fs=fs
    ## istft is now in torch
    #audio = torchaudio.functional.istft(stft_matrix = complex_ri, n_fft=int(1024*fs), hop_length=int(256*fs), win_length=int(1024*fs), window=window, center=True, pad_mode='reflect', normalized=False, onesided=True, length=length)

    audio = torch.istft(complex_ri, n_fft=int(1024*fs), hop_length=int(256*fs), win_length=int(1024*fs), window=window, center=True,  normalized=False, onesided=True, length=length)
    #audio = audio.numpy().squeeze()
    return audio

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
    
def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                li.append(os.path.join(os.path.abspath(d_name), filename))
    len_li = len(li)            
    return li

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-m','--model',type=str,default='./model_ckpt/bestmodel.pth')
    parser.add_argument('-i','--input_dir',type=str,required=True)
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config)
    print('Using configuration :: ' + args.config)

    device = hp.gpu
    torch.cuda.set_device(device)
    
    fs = hp.train.fs/16 #16,32,48
    orig_fs = hp.train.fs/16
    batch_size = 1
    target_fs =[1,2,3] #16,32,48
    
    if fs!=1 and fs!=2 and fs!=3:
        re_fs = find_nearest(target_fs,fs)
        win_len = 1024*re_fs
        window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)
    else:
        re_fs = fs
        win_len = 1024*fs
        window=torch.hann_window(window_length=int(win_len), periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False).to(device)

    test_model = args.model
    num_epochs = 1

    #data_tesnt = args.input_dir
    #data_test_list=[]
    #data_test_list=search(data_test,data_test_list)

    #data_test_list = [x for x in glob.glob(os.path.join(args.input_dir,'**'),recursive=True) if not os.path.isdir(x)]
    # temp
    data_test_list = [x for x in glob.glob(os.path.join(args.input_dir, '*_simu', '*CH5.wav'),recursive=True) if not os.path.isdir(x)] + [x for x in glob.glob(os.path.join(args.input_dir, '*_real', '*CH5.wav'),recursive=True) if not os.path.isdir(x)]




    #print('data_test_list' + str(data_test_list))

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_dataset = testDataset(args.input_dir,data_test_list,re_fs,orig_fs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, collate_fn=lambda x:my_collate(x),shuffle=False,num_workers=8)
    model_test = UNet().to(device)
    print('Loading Model : ' + test_model)
    model_test.load_state_dict(torch.load(test_model,map_location=device))
    model_test.eval()
    
    with torch.no_grad():
        for i, (data_dir,data_name,data_wav_len,re_fs,data_wav,input_wav_real,input_wav_imag) in enumerate(tqdm(test_loader)):
            audio_real = input_wav_real.to(device)
            audio_imagine = input_wav_imag.to(device)
            audio_maxlen = int(audio_real.shape[-1]*256*fs-1)
            
            mask_r,mask_i = model_test(audio_real,audio_imagine)
            if hp.train.type == 'R':
                enhance_r = audio_real * mask_r
                enhance_i = audio_imagine * mask_i
            else :
                enhance_r = audio_real * mask_r - audio_imagine * mask_i
                enhance_i = audio_real * mask_i + audio_imagine * mask_r

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            
            if hp.eval.type =='cplx' : 
                enhance_spec = torch.cat((enhance_r,enhance_i),3)
                audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,re_fs)
            # Use enhanced magnitude and original phase
            else  : 
                enhance_spec = torch.cat((enhance_r,enhance_i),3)

                ## magphase()
                # Separate a complex-valued spectrogram with shape (…, 2) into its magnitude and phase.
                enhance_mag,enhance_phase = torchaudio.functional.magphase(enhance_spec)

                audio_real = audio_real.unsqueeze(3)
                audio_imagine = audio_imagine.unsqueeze(3)

                original_spec = torch.cat((audio_real,audio_imagine),3)

                original_mag,original_phase = torchaudio.functional.magphase(original_spec)
                estimated_spec = enhance_mag * torch.exp(1j*original_phase)
                
                # torch.view_as_real : since pytorch 1.6
                estimated_spec = torch.view_as_real(estimated_spec)
                audio_me_pe = complex_demand_audio(estimated_spec,window,audio_maxlen,re_fs)

            data_name = data_name[0]
            re_sr = re_fs
            audio_me_pe=audio_me_pe.to('cpu')

            max_data_wav = data_wav.max()
            min_data_wav = data_wav.min()
            if abs(max_data_wav) >= abs(min_data_wav):
                norm_data = abs(min_data_wav)
            else:
                norm_data = abs(max_data_wav)
            if audio_me_pe.max() >=1 or audio_me_pe.min() <=-1:
                max_aud = audio_me_pe.max()
                min_aud = audio_me_pe.min()
                if abs(max_aud) >= abs(min_aud):
                    audio_me_pe = audio_me_pe * (norm_data/max_aud)
                else:
                    audio_me_pe = audio_me_pe * (norm_data/abs(min_aud))
            
            data_dir = data_dir[0]
            if not os.path.exists(output_dir +'/'+ data_dir):
                os.makedirs(output_dir +'/'+ data_dir)
            torchaudio.save(output_dir+"/"+data_dir+'/'+data_name+".wav",src=audio_me_pe[:,:int(data_wav_len)], sample_rate=int(16000*re_sr))
            



