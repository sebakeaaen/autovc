# %%
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc_mel import Generator
import matplotlib.pyplot as plt
from librosa import display

print('Started conversion')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id = 'chkpnt_spmel_reproducedAutoVC_new_22April23_1444_43_resumed_resumed_resumed_resumed'#'autovc' #checkpoint
model_type = 'spmel'

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load(id+'.ckpt', map_location=device)
G.load_state_dict(g_checkpoint['state_dict']) #state_dict for our models

metadata = pickle.load(open('ninatest/metadata.pkl', "rb"))

spect_vc = []

for conversion in metadata:
    #FROM:         
    x_org = conversion[1][2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(conversion[1][1][np.newaxis, :]).to(device)

    display.specshow(
        x_org.T * 100 - 100,
        y_axis="mel",
        x_axis="time",
        fmin=90,
        fmax=7_600, 
        sr=16_000,
    )
    plt.savefig('ninatest/'+str(conversion[0])+'_original_mel.pdf')

    #TO:               
    emb_trg = torch.from_numpy(conversion[2][1][np.newaxis, :]).to(device)
    #print('input shapes')
    #print(uttr_org.shape)
    #print(emb_org.shape)
    #print(emb_trg.shape)

    
    
    with torch.no_grad():
        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
        
    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    display.specshow(
        uttr_trg.T * 100 - 100,
        y_axis="mel",
        x_axis="time",
        fmin=90,
        fmax=7_600, 
        sr=16_000,
    )
    plt.savefig('ninatest/'+str(conversion[0])+'_translation_mel.pdf')
    
    #carry the filename/conversion identifier in the metadata.log file forward to the vocoder which will create the
    spect_vc.append( (f'{str(conversion[0])}', uttr_trg) )
        
        
with open('ninatest/results_'+id+'.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)

print('Finished conversion...')      


