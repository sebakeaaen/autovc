
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

pre_trained = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)

if pre_trained:
    c_checkpoint = torch.load('3000000-BL.ckpt', map_location = device)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).to(device)
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

# Our modification: Reading numpy files 
with open(r"spmel/train.pkl", "rb") as file:
    train = pickle.load(file)

#train_cpy = train
#for subject in train_cpy:
#    for i, np_file in enumerate(subject[2:]):
#        i=i+2
#        subject[i] = np.load('spmel\\'+np_file)

train_cpy = train
for subject in train_cpy:
    subject[3:] = [] # remove the npy don't need
    for i in range(0,len(subject[2:])):
        np_file = subject[2:][0] # 0 is the index of the chosen npy file
        subject[2] = np.load('spmel/'+np_file)
        
with open(os.path.join('.', 'metadata.pkl'), 'wb') as handle:
    pickle.dump(train_cpy, handle)