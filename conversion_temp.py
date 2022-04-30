# %%
import os
import pickle
from re import sub
import torch
import numpy as np
from math import ceil
from model_vc_mel import Generator as mel_Generator
from model_vc_stft import GeneratorSTFT as stft_Generator
from model_vc_wav import GeneratorWav as wav_Generator
import pandas as pd

##choose type and modify id below according to name of model checkpoint
model_type = 'spmel'
id = {
      'spmel':'chkpnt_spmel_reproducedAutoVC_new_22April23_1444_43_resumed_resumed_resumed_resumed',
       'stft': 'chkpnt_stft_stft_scratch_22April21_1408_13_resumed_resumed_resumed',
        'wav': 'add_checkpoint_name_here'
      }

subject_conversions = [
                        (('p226','001'),('p226','001')), 
                        (('p226','003'),('p363','003'))
                        ]


main_dir = '/work3/dgro/VCTK-Corpus-0/' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
speaker_info = pd.read_csv('speaker_info.txt', delim_whitespace=True)


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

if model_type == 'spmel':
    G = mel_Generator(32,256,512,32).eval().to(device)
elif model_type == 'stft':
    G = stft_Generator(32,256,512,32).eval().to(device)
elif model_type == 'wav':
    G = wav_Generator(32,256,512,32).eval().to(device) 


g_checkpoint = torch.load(os.path.join(main_dir,'models',str(id[model_type])+'.ckpt'), map_location=device)
G.load_state_dict(g_checkpoint['state_dict'])
metadata = pickle.load(open('/work3/dgro/VCTK-Corpus-0/' + model_type +'/metadata.pkl', "rb"))

#metadata_dict = {}

#for subject in metadata:
#    metadata_dict['id'] = subject[0]


print('loaded checkpoint and metadata...')

print('Starting results.pkl generation...')

for conversion in subject_conversions:
    print('#'*40+'\n')
    with open(os.path.join(main_dir,'txt',conversion[0][0],conversion[0][0] + '_' + conversion[0][1]+'.txt',), 'r') as sentence_file:
        sentence = "\"" + sentence_file.readline().rstrip('\n').rstrip() + "\""
        print(f'Converting from sentence no. {conversion[0][1]} : {sentence}')
        
    print('Uttered by the speaker:')
    print(speaker_info[speaker_info['ID']==conversion[0][0]].to_string(index=False))
    print('')

    with open(os.path.join(main_dir,'txt',conversion[1][0],conversion[1][0] + '_' + conversion[1][1]+'.txt',), 'r') as sentence_file:
        sentence = "\"" + sentence_file.readline().rstrip('\n').rstrip() + "\""
        print(f'Into sentence no. {conversion[1][1]}: {sentence}')

    print('Uttered by the speaker:')
    print(speaker_info[speaker_info['ID']==conversion[1][0]].to_string(index=False))
    print('\n')



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


