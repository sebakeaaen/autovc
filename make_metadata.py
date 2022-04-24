"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

class Metadata(object):

    def __init__(self, config):
        """Initialize configurations."""

        self.speaker_embed = config.speaker_embed
        self.main_dir = config.main_dir
        self.model_type = config.model_type
        self.root_dir = self.main_dir+'/'+self.model_type # containing spmel or stft spects
    
    def metadata(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
        
        if self.speaker_embed:
            c_checkpoint = torch.load('3000000-BL.ckpt',map_location=device)
            new_state_dict = OrderedDict()
            for key, val in c_checkpoint['model_b'].items():
                new_key = key[7:]
                new_state_dict[new_key] = val
            C.load_state_dict(new_state_dict)

        num_uttrs = 10
        len_crop = 128
        
        # speaker embedding is created from mel! (not stft, since checkpoint 300000-BL.pth is created for mel)
        self.mel_dir = self.main_dir+'/'+'spmel'
        dirName, subdirList, _ = next(os.walk(self.mel_dir))
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
            if self.speaker_embed:
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
                utterances.append(np.mean(embs, axis=0)) # average speaker embedding
            else: # one-hot encoding
                one_hot_encoding[i] = 1
                utterances.append(one_hot_encoding)
                i += 1
    
            # create file list
            for fileName in sorted(fileList):
                utterances.append(os.path.join(speaker,fileName))
            speakers.append(utterances)
    
        with open(os.path.join(self.root_dir, 'train.pkl'), 'wb') as handle:
            pickle.dump(speakers, handle)

        ######### Our modification: Reading numpy files in speaker embedding ##########
        if self.speaker_embed: # i.e. not neccessesary for one-hot encoding..
            with open(os.path.join(self.root_dir, 'train.pkl'), 'rb') as file:
                train = pickle.load(file)

            #create metadata for testing
            #Format is [subject, embedding, either mel or stft spectogram, sentence numpy filename] 
            #          [str, (256,), either (x,80) or (x,513), str]
            metadata = []
            for subject in train:
                first_mel_spec = np.load(self.root_dir+'/'+subject[2])
                metadata.append(subject[0:2] + [first_mel_spec] + [subject[2]])
        
            with open(os.path.join(self.root_dir, 'metadata.pkl'), 'wb') as handle:
                pickle.dump(metadata, handle)

