# %%
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc_mel import Generator
from model_vc_stft import GeneratorSTFT
import matplotlib.pyplot as plt
from librosa import display
import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import librosa
from numpy.random import RandomState
from sklearn.preprocessing import RobustScaler

cutoff = 30
fs = 16000
order = 5
fft_length = 1024
hop_length = 256
n_fft = 1024
n_mels = 128

mel_basis = mel(fs, n_fft, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

print('Started conversion')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id = 'chkpnt_stft_stft_scratch_22April21_1408_13_resumed_resumed'
model_type = 'stft_test'
main_dir = '/work3/dgro/VCTK-Corpus-0'

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

if model_type=='spmel':
    G = Generator(32,256,512,32).eval().to(device)
else:
    G = GeneratorSTFT(32,256,512,32).eval().to(device)
g_checkpoint = torch.load(main_dir+'/models/'+id+'.ckpt', map_location=device)

G.load_state_dict(g_checkpoint['state_dict']) #state_dict for our models

metadata = pickle.load(open(main_dir + '/' + model_type +'/' + 'metadata.pkl', "rb"))

spect_vc = []

for conversion in metadata:
    #FROM:         
    x_org = conversion[1][2]
    x_org, len_pad = pad_seq(x_org)

    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(conversion[1][1][np.newaxis, :]).to(device)

    uttr_org_mel = uttr_org[0, :-len_pad, :].cpu().numpy()
    uttr_org_mel = np.dot(uttr_org_mel, mel_basis)

    print(uttr_org_mel.shape)


    display.specshow(
        uttr_org_mel.T * 100 - 100,
        y_axis=("mel"),
        x_axis="time",
        fmin=90,
        fmax=7_600, 
        sr=16_000,
    )
    plt.savefig(main_dir + '/' + model_type +'/'+str(conversion[0])+'_original_mel.pdf')

    #TO:               
    emb_trg = torch.from_numpy(conversion[2][1][np.newaxis, :]).to(device)
    #print('input shapes')
    #print(uttr_org.shape)
    #print(emb_org.shape)
    #print(emb_trg.shape)

    
    if model_type=='spmel':
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
    else:
        with torch.no_grad():
            x_identic_psnt, _, _ = G(uttr_org, emb_org, emb_trg)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()

    uttr_trg_mel = np.dot(uttr_trg, mel_basis)

    display.specshow(
        uttr_trg_mel.T * 100 - 100,
        y_axis=("mel"),
        x_axis="time",
        fmin=90,
        fmax=7_600, 
        sr=16_000,
    )
    plt.savefig(main_dir + '/' + model_type +'/'+str(conversion[0])+'_translation_mel.pdf')

    print(uttr_trg_mel.shape)
    
    #carry the filename/conversion identifier in the metadata.log file forward to the vocoder which will create the
    spect_vc.append( (f'{str(conversion[0])}', uttr_trg_mel) )
        
        
with open(main_dir + '/' + model_type +'/' + 'results_'+id+'.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)

print('Finished conversion...')      


