"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
import pandas as pd

class Metadata(object):

    def __init__(self, config):
        """Initialize configurations."""

        self.speaker_embed = config.speaker_embed
        self.main_dir = config.main_dir
        self.model_type = config.model_type
        self.root_dir = self.main_dir+'/'+self.model_type # containing spmel or stft spects
        self.num_uttrs = 10
        #we use spmel for metadata speaker encoding regardless
        self.len_crop = 128

        self.subject_conversions = [ #((original speaker, sentence), target speaker)
                        (('p002','001'),'p226'),
                        (('p226','001'),'p002')
                        ]
        self.main_dir = config.main_dir

        self.speaker_info = pd.read_csv('speaker_info.txt', delim_whitespace=True)
    
    def metadata(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
        
        c_checkpoint = torch.load('3000000-BL.ckpt',map_location=device)
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        C.load_state_dict(new_state_dict)

        num_uttrs = self.num_uttrs
        len_crop = self.len_crop
        
        # speaker embedding is created from mel! (not stft, since checkpoint 300000-BL.pth is created for mel)
        self.mel_dir = self.main_dir+'/spmel'
        dirName, subdirList, _ = next(os.walk(self.mel_dir))
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
                melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
                emb = C(melsp)
                embs.append(emb.detach().squeeze().cpu().numpy())     
            utterances.append(np.mean(embs, axis=0)) # average speaker embedding
    
            # create file list
            for fileName in sorted(fileList):
                utterances.append(os.path.join(speaker,fileName))
            speakers.append(utterances)
    
        #with open(os.path.join(self.root_dir, 'train.pkl'), 'wb') as handle:
        #    pickle.dump(speakers, handle)
        with open('train.pkl', 'wb') as handle:
            pickle.dump(speakers, handle)

        ######### Our modification: Reading numpy files in speaker embedding ##########
        #with open(os.path.join(self.root_dir, 'train.pkl'), 'rb') as file:
        with open('train.pkl', 'rb') as file:
            train = pickle.load(file)
        #construct hash map for easy retrieval    
        subject_speaker_embedding = {}
        for embed in train:
            subject_speaker_embedding[embed[0]] = embed[1]


        #with open(os.path.join(self.root_dir, 'metadata.log'), 'w') as log:
        with open('metadata.log', 'w') as log:
            log_ref_int = 0
            metadata = [] #format is list of conversion metadata [log ref int(eventual filename), [from subject.sentence, from embedding, from sound input], [to subject, to embedding]
            for conversion in self.subject_conversions:
                log.write('CONVERSION FILENAME: '+str(log_ref_int) +' ' + '#'*40 + '\n\n')
                with open(os.path.join(self.main_dir,'txt',conversion[0][0],conversion[0][0] + '_' + conversion[0][1]+'.txt',), 'r') as sentence_file:
                    sentence = "\"" + sentence_file.readline().rstrip('\n').rstrip() + "\""
                    log.write(f'Converting from sentence no. {conversion[0][1]} : {sentence} \n')
                    
                log.write('Uttered by the speaker:\n')
                log.write(self.speaker_info[self.speaker_info['ID']==conversion[0][0]].to_string(index=False))
                log.write('\n')

                log.write('To the speaker:\n')
                log.write(self.speaker_info[self.speaker_info['ID']==conversion[1]].to_string(index=False))
                log.write('\n\n')

                #bad use for try/except, checking for mic2. I DON'T CARE. YOLO

                try:
                    sound_input = np.load(self.root_dir+'/'+conversion[0][0] + '/' + conversion[0][0] + '_' + conversion[0][1] + '_mic2.npy')
                except:
                    sound_input = np.load(self.root_dir+'/'+conversion[0][0] + '/' + conversion[0][0] + '_' + conversion[0][1] + '.npy')

                metadata.append([log_ref_int, 
                                [conversion[0][0]+'_'+conversion[0][1], subject_speaker_embedding[conversion[0][0]], sound_input],
                                [conversion[1], subject_speaker_embedding[conversion[1]]]
                                ])

                log_ref_int = log_ref_int + 1

            #with open(os.path.join(self.root_dir, 'metadata.pkl'), 'wb') as handle:
            with open('metadata.pkl', 'wb') as handle:
                pickle.dump(metadata, handle)
        print('Finished generating metadata')