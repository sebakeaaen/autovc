import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn
from make_metadata import Metadata
from make_spect import Spect
from datetime import datetime


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Generate spectrogram data from the wav files (if not already done)
    if  os.path.exists(os.path.join(config.main_dir, config.model_type)) == False:
        print('Did not find folder with spectrograms - creating...')
        sp = Spect(config)
        sp.spect()
    else:
        print('Found folder with spectrograms - continuing...')

    # Generate training metadata (if not allready done)
    path = config.main_dir + '/' + config.model_type + '/train.pkl'
    if os.path.exists(path):
        print("Metadata already created - continuing...")
    else:
        print('Metadata does not exists - creating...')
        md = Metadata(config)
        md.metadata()

    # Data loader.
    vcc_loader = get_loader(config.main_dir, config.batch_size, config.len_crop, config.model_type)
    
    solver = Solver(vcc_loader, config)

    solver.train()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss, L_content')
    parser.add_argument('--lambda_SISNR', type=float, default=1, help='weight for L_SISNR')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256) # if one-hot encoding, change to no. of subjects
                                                            # 110 subjects in total
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    
    # Training configuration.
    parser.add_argument('--main_dir', type=str, default='/work3/dgro/VCTK-Corpus-0') # consider if train should be on all or only subset
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=10000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length: 128 for mel and stft, (127*256)+1024=33536 for wav')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--speaker_embed',type=bool, default=True, help='mel-based speaker embedding or one-hot-encoding')
    parser.add_argument('--model_type',type=str,default='stft',help='input/output type: spmel, stft or wav')
    parser.add_argument('--run_name',required=True, type=str, help='name of run for wandb and checkpoints')
    parser.add_argument('--lr_scheduler',type=str,default=None,help='Cosine or Plateau')
    parser.add_argument('--depth',type=int,default=1,help='ConvTasNet encoder and decoder depth: 1, 3 or 5 layers')
    parser.add_argument('--ema', type=float, default=0.9999, help='Exponential moving average weight')
    parser.add_argument('--resume',type=bool,default=False, help='resume training from checkpoint')
    parser.add_argument('--run_id', type=str, default=None, help='wandb run id (see run overview in wandb: run path, e.g. 1g7y0hr9)')

    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=100)

    config = parser.parse_args()

    if not config.resume:
        config.run_name = config.run_name + datetime.now().strftime('_%y%B%d_%H%M_%S')
    else:
        print('Resuming training..')
        # Note: if resuming training, config.run_name should be the run name of the initial run

    print(config)
    main(config)