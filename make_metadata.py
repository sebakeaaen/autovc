"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle

from importlib_metadata import metadata
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import argparse

np.random.seed(42069) # this works for all subjects in training folder!

class Metadata(object):

    def __init__(self, config):
        """Initialize configurations."""

        self.speaker_embed = config.speaker_embed
        self.model_type = config.model_type
        self.rootDir = config.data_dir+'/'+config.model_type # Directory containing spectrograms
        self.rootDirMel = config.data_dir+'/spmel'
    
    def metadata(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.speaker_embed:
            C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
            c_checkpoint = torch.load('3000000-BL.ckpt',map_location=device)
            new_state_dict = OrderedDict()
            for key, val in c_checkpoint['model_b'].items():
                new_key = key[7:]
                new_state_dict[new_key] = val
            C.load_state_dict(new_state_dict)

        num_uttrs = 10
        len_crop = 128

        dirName, subdirList, _ = next(os.walk(self.rootDir))
        print('Found directory: %s' % dirName)

        dirNameMel, _, _ = next(os.walk(self.rootDirMel))

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
                tmp = np.load(os.path.join(dirNameMel, speaker, fileList[idx_uttrs[i]]))
                candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
                # choose another utterance if the current one is too short
                while tmp.shape[0] < len_crop:
                    idx_alt = np.random.choice(candidates)
                    tmp = np.load(os.path.join(dirNameMel, speaker, fileList[idx_alt]))
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
    
        with open(os.path.join(self.rootDir, 'train.pkl'), 'wb') as handle:
            pickle.dump(speakers, handle)

        ######### Our modification: Reading numpy files in speaker embedding ##########
        if self.speaker_embed:
            with open(os.path.join(self.rootDir, 'train.pkl'), 'rb') as file:
                train = pickle.load(file)

            #create metadata for testing
            #Format is [subject, embedding, melspec] 
            metadata = []
            for subject in train:
                first_mel_spec = np.load(self.rootDir+'/'+subject[2])
                metadata.append(subject[0:2] + [first_mel_spec])
        
            #with open(os.path.join('.', 'metadata.pkl'), 'wb') as handle:
            with open(os.path.join(self.rootDir, 'metadata.pkl'), 'wb') as handle:
                pickle.dump(metadata, handle)
