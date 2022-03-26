import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn
from make_metadata import Metadata
from make_spect import Spect
import glob


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Generate spectrogram data from the wav files (if not already done)
    #path = config.data_dir + '/' + config.model_type # check if path is empty
    if  os.path.exists(os.path.join(config.data_dir, config.model_type)) == False:
        print('Did not find folder with spectrograms - creating...')
        sp = Spect(config)
        sp.spect()
    else:
        print('Found folder with spectrograms - continuing...')

    # Generate training metadata (if not allready done)
    #path = config.data_dir+'/'+config.model_type+'/metadata.pkl'
    #if len(os.listdir(path)) == 0: # check if path is empty
    path = config.data_dir + '/' + config.model_type + '/train.pkl'
    if os.path.exists(path):
        print("Metadata already created - continuing...")
    else:
        print('Metadata does not exists - creating...')
        md = Metadata(config)
        md.metadata()

    # Data loader.
    vcc_loader = get_loader(config.data_dir, config.batch_size, config.len_crop, config.model_type)
    
    solver = Solver(vcc_loader, config)

    solver.train()
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=16)
    parser.add_argument('--dim_emb', type=int, default=256) # if one-hot encoding, change to no. of subjects
                                                            # 110 subjects in total
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='/work3/dgro/VCTK-Corpus-0') # consider if train should be on all or only subset
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--speaker_embed',type=bool, default=True, help='mel-based speaker embedding or one-hot-encoding')
    parser.add_argument('--model_type',type=str,default='spmel',help='input/output type: spmel or stft')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    print(config)
    main(config)