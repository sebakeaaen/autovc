# %%
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#id = 'ditterun_22March31_1545_40'
id = 'ditte_22April06_1003_01'
model_type = 'spmel'

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

#G = Generator(32,256,512,32).eval().to(device) # if speaker embedding
G = Generator(32,256,512,32, speaker_embed=True).eval().to(device) # if one-hot encoding (110 is the number of subjects)

g_checkpoint = torch.load('chkpnt_'+model_type+'_'+id+'.ckpt', map_location=device)
G.load_state_dict(g_checkpoint['state_dict'])

#metadata = pickle.load(open('/work3/dgro/VCTK-Corpus-0/' + model_type +'/metadata.pkl', "rb"))
metadata = pickle.load(open('metadata.pkl', "rb"))

spect_vc = []

for sbmt_i in metadata:
             
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
with open('results_'+id+'.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)          


