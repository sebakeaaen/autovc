"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

pre_trained = False # OBS! change in conversion.ipynb, vocoder.ipynb and make_metadata.py
if pre_trained:
    speaker_embedding = True
else:
    speaker_embedding = False # if false, one-hot encoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#if speaker_embedding == False:
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)

if pre_trained == True:
    c_checkpoint = torch.load('3000000-BL.ckpt',map_location=device)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)


num_uttrs = 10
len_crop = 128

# Directory containing spectrograms
rootDir = '/work3/dgro/VCTK-Corpus-0/STFT'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
i = 0 # for the one-hot encoding

for speaker in sorted(subdirList):
    one_hot_encoding = torch.zeros(len(subdirList))
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    if speaker_embedding:
        for i in range(num_uttrs):
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
            left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())     
        utterances.append(np.mean(embs, axis=0))
    else:
        # one-hot encoding
        one_hot_encoding[i] = 1
        utterances.append(one_hot_encoding)
        i += 1
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)


######### Our modification: Reading numpy files ##########
# with open(r"spmel/train.pkl", "rb") as file:
#     train = pickle.load(file)

# # multiple sentences (for training?)
# #train_cpy = train
# #for subject in train_cpy:
# #    for i, np_file in enumerate(subject[2:]):
# #        i=i+2
# #        subject[i] = np.load('spmel/'+np_file)

# # single sentence (for conversion)
# train_cpy = train
# for subject in train_cpy:
#     subject[3:] = [] # remove the npy don't need
#     for i in range(0,len(subject[2:])):
#         np_file = subject[2:][0] # 0 is the index of the chosen npy file
#         subject[2] = np.load('spmel/'+np_file)
        
# with open(os.path.join('.', 'metadata.pkl'), 'wb') as handle:
#     pickle.dump(train_cpy, handle)
