from torch.utils import data
import torch
import numpy as np
import pickle 
import os
import torch.nn.functional as F 
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, data_dir, len_crop, model_type):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = data_dir + '/' + model_type
        self.len_crop = len_crop
        self.step = 10

        torch.cuda.empty_cache()
        
        with open(os.path.join(self.root_dir, 'train.pkl'), 'rb') as file:
            meta = pickle.load(file)
        
        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(len(self.train_dataset)):
            for j in range(len(self.train_dataset[i])):
                if j > 0:
                    self.train_dataset[i][j] = torch.from_numpy(self.train_dataset[i][j])#.to(self.device)
                    
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join(self.root_dir, tmp))
            dataset[idx_offset+k] = uttrs
                   
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        tmp = list_uttrs[a].to(self.device)
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            #uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant') # change to torch padding
            uttr = torch.nn.functional.pad(tmp, (0, 0, 0, len_pad), "constant")
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            uttr = tmp[left:left+self.len_crop, :]
        else:
            uttr = tmp
        
        return uttr, emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(root_dir, batch_size=16, len_crop=128, model_type = 'spmel', num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, len_crop, model_type)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader