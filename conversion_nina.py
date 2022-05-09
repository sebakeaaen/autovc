# %%
from bdb import set_trace
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc_mel import Generator
from model_vc_stft import GeneratorSTFT
from model_vc_wav import GeneratorWav
import matplotlib.pyplot as plt
from librosa import display
from librosa.filters import mel
from scipy.signal import get_window
from sklearn.preprocessing import RobustScaler

cutoff = 30
fs = 16000
order = 5
fft_length = 1024
hop_length = 256
n_fft = 1024
n_mels = 80
    
def pySTFT(x):

    x = np.pad(x, int(fft_length//2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                            strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.abs(np.fft.rfft(fft_window * result, n=fft_length).T) #inverse function is irfft 
    return result

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

print('Started conversion')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id = 'chkpnt_wav_nina_wav5_scale0.1_22May03_1543_01_ditte'#autovc #checkpoint
model_type = 'wav'
depth = 5

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

if model_type=='spmel':
    G = Generator(32,256,512,32).eval().to(device)
if model_type =='stft':
    G = GeneratorSTFT(32,256,512,32).eval().to(device)
if model_type == 'wav':
    G = GeneratorWav(32,256,512,32,depth).eval().to(device)

g_checkpoint = torch.load('/work3/dgro/VCTK-Corpus-0/models/'+id+'.ckpt', map_location=device)
G.load_state_dict(g_checkpoint['state_dict']) #state_dict for our models

path = '/work3/dgro/VCTK-Corpus-0/' + model_type

metadata = pickle.load(open(path+'/metadata.pkl', "rb"))

spect_vc = []

for conversion in metadata:
    #FROM:
    x_org = conversion[1][2]
    x_org = x_org[:33536,:]
    if model_type == 'spmel':         
        x_org = x_org
    elif model_type == 'stft':
        if x_org.shape[0] == 513:
            x_org = conversion[1][2].T
        else:
            x_org = conversion[1][2]

    print(x_org.shape)

    x_mel = np.copy(x_org)
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(conversion[1][1][np.newaxis, :]).to(device)

    print(mel_basis.shape)

    if model_type == 'stft':
        x_mel = x_mel
        D_mel = np.dot(x_mel, mel_basis)
    elif model_type == 'wav':
        x_mel = pySTFT(x_mel.squeeze())
        print(x_mel.shape)
        D_mel = np.dot(x_mel.T, mel_basis)

    D_mel = 20 * np.log10(np.maximum(min_level, D_mel)) - 16

    print(D_mel.shape)

    display.specshow(
        (D_mel.T*100-100),
        y_axis="mel",
        x_axis="s",
        fmin=90,
        fmax=7_600, 
        sr=16_000,
        hop_length = 256
    )
    plt.savefig(path+'/'+str(conversion[0])+'_'+str(model_type)+str(depth)+'_original_mel.pdf')

    plt.close()

    #TO:               
    emb_trg = torch.from_numpy(conversion[2][1][np.newaxis, :]).to(device)
    #print('input shapes')
    #print(emb_org.shape)
    #print(emb_trg.shape)
    
    with torch.no_grad():
        if model_type == 'wav':
            _, x_identic_psnt, _, _ = G(uttr_org, emb_org, emb_trg)
        else:
            x_identic, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            x_identic_psnt = x_identic_psnt.squeeze(0)
            x_identic = x_identic.squeeze(0)

    print(x_identic_psnt.shape)
    if model_type == 'stft':
        x_identic_psnt = x_identic
  
    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        #x_identic = x_identic[0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
        #x_identic = x_identic[0, :-len_pad, :].cpu().numpy()

    if (model_type in ['stft']):
       D_mel_trg = np.dot(uttr_trg, mel_basis)
    elif model_type =='wav':
        D_mel = pySTFT(uttr_trg.squeeze())
        D_mel_trg = np.dot(D_mel.T, mel_basis)
    else:
        D_mel_trg = uttr_trg.squeeze()
    

    D_mel_trg = 20 * np.log10(np.maximum(min_level, D_mel_trg)) - 16

    # display.specshow(
    #     (x_identic.T * 100 -100),
    #      y_axis="mel",
    #      x_axis="s",
    #      fmin=90,
    #      fmax=7_600, 
    #      sr=16_000,
    #      hop_length = 256
    #  )
    
    # plt.savefig(path+'/'+str(conversion[0])+'_'+str(model_type)+str(depth)+'_conversion_mel.pdf')

    # plt.close()

    print(D_mel_trg.shape)

    display.specshow(
         (D_mel_trg.T*100-100),
         y_axis="mel",
         x_axis="s",
         fmin=90,
         fmax=7_600, 
         sr=16_000,
         hop_length = 256
     )
    plt.savefig(path+'/'+str(conversion[0])+'_'+str(model_type)+str(depth)+'_conversion_post_mel.pdf')

    plt.close()
    
    #carry the filename/conversion identifier in the metadata.log file forward to the vocoder which will create the
    spect_vc.append( (f'{str(conversion[0])}', D_mel_trg) )
        
        
with open(path+'/results_'+id+'.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)

print('Finished conversion...')      



# %%
