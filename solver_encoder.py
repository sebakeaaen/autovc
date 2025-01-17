from functools import lru_cache
from syslog import LOG_SYSLOG
from model_vc_mel import Generator 
from model_vc_stft import GeneratorSTFT
from model_vc_wav import GeneratorWav
import torch
import torch.nn.functional as F
import time
import datetime
import wandb
import matplotlib.pyplot as plt
from librosa import display
from librosa.feature import melspectrogram
from scipy.signal import get_window
from librosa.filters import mel
import numpy as np
import os
from sisdr_loss import SingleSrcNegSDR

cutoff = 30
fs = 16000
order = 5
fft_length = 1024
hop_length = 256
n_fft = 1024
n_mels = 128

def pySTFT(x):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                            strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.abs(np.fft.rfft(fft_window * result, n=fft_length).T) #inverse function is irfft 
    return result

def plot_mel(x):
            mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
            D = pySTFT(x).T
            D_mel = np.dot(D, mel_basis)
            min_level = np.exp(-100 / 20 * np.log(10))
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = np.clip((D_db + 100) / 100, 0, 1)
            S_temp = (S[0]).T
            return S_temp

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        self.main_dir = config.main_dir

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_SISNR = config.lambda_SISNR
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.lr = config.lr
        self.lr_scheduler = config.lr_scheduler
        self.depth = config.depth

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.ema = config.ema
        self.run_name = config.run_name
        self.resume = config.resume
        self.run_id = config.run_id
        
        # models
        self.model_type = config.model_type
        self.speaker_embed = config.speaker_embed
        
        self.log_step = config.log_step

        #wandb setup
        with open('wandb.token', 'r') as file:
            api_key = file.readline()
            wandb.login(key=api_key)

        self.path = 'chkpnt_' + self.model_type + '_' + self.run_name + '.ckpt'
        self.file_exists = os.path.exists(self.path)

        if self.file_exists:
            wandb.init(project="DNS autovc", entity="macaroni", config=config, reinit=True, id=self.run_id, resume=True)
        else:
            wandb.init(project="DNS autovc", entity="macaroni", config=config, reinit=True, name=self.run_name)
        
        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("Training on GPU.")
        else:
            print("Training on CPU.")
            wandb.alert(
                            title=f"Training on {self.device}", 
                            text=f"Training on {self.device}"
                        )

        # Build the model and tensorboard.
        self.build_model()

        # Set up weights and biases config
        #wandb.config.update(config, allow_val_change=True)


    def build_model(self):
        
        if self.model_type == 'spmel':
            self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        elif self.model_type == 'stft':
            self.G = GeneratorSTFT(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)  
        elif self.model_type == 'wav':
            self.G = GeneratorWav(self.dim_neck, self.dim_emb, self.dim_pre, self.freq, self.depth)
        else: print('Model type not recognized')

        self.G.to(self.device)

        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.lr)
        '''
        # Using different learning rates for different model layers
        self.g_optimizer = torch.optim.Adam([
            {"params": self.G.ConvTasEncoder.parameters(), "lr": self.lr_convtas},
            {"params": self.G.ConvTasDecoder.parameters(), "lr": self.lr_convtas}],
            lr=self.lr_global)
        '''

        if self.lr_scheduler == 'Cosine': 
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=10000, eta_min=0)
        elif self.lr_scheduler == 'Plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
        else:
            print('No learning rate scheduler used.')
            self.lr_scheduler = None

        if self.file_exists:
            print('Loading checkpoint: chkpnt_' + self.model_type + '_' + self.run_name + '.ckpt')
            checkpoint = torch.load(self.path, map_location=self.device)
            self.G.load_state_dict(checkpoint['state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['optimizer'])
            self.i = checkpoint['epoch']
            self.loss = checkpoint['loss']

            ''' 
            # not neccessary
            # manually moving optimizer state to GPU 
            for state in self.g_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            '''
        
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def model_EMA(self):
        # compute Exponential Moving Average (EMA) weights
        flat_params = torch.cat([param.data.view(-1) for param in self.G.parameters()], 0)
        avg_params = self.ema * flat_params + (1-self.ema) * flat_params

        # overwrite model weights with EMA weights
        offset = 0
        for param in self.G.parameters():
            param.data.copy_(avg_params[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
    
    #=====================================================================================================================================#
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        lr = self.lr
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

        if self.file_exists:
            i_start = self.i
            print('Continue from iteration: ',i_start)
        else:
            i_start = 0

        # Start training.
        print('Starting training...')
        start_time = time.time()

        self.G.train()

        wandb.watch(self.G, log = None)

        for i in range(i_start, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)

            x_real = x_real.to(self.device)
            emb_org = emb_org.to(self.device)
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
                        
            # Identity mapping loss
            if self.model_type in ('spmel','stft'):
                x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
                # L_recon
                g_loss_id = F.mse_loss(x_real.squeeze(), x_identic.squeeze()) 

                # L_recon0
                g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())  

                code_reconst = self.G(x_identic_psnt, emb_org, None) 
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                # L_SISNR: SI-SNR loss
                # not used for mel model
                g_loss_SISNR = torch.tensor(float('nan'))

                # Total loss
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd

            # elif self.model_type == 'stft':
            #     x_identic, code_real = self.G(x_real, emb_org, emb_org)
            #     # L_recon
            #     g_loss_id = F.mse_loss(x_real.squeeze(), x_identic.squeeze()) 

            #     code_reconst = self.G(x_identic, emb_org, None) 
            #     g_loss_cd = F.l1_loss(code_real, code_reconst)

            #     # L_recon0
            #     # set postnet loss to nan (since we found out that postnet makes no difference)
            #     g_loss_id_psnt = torch.tensor(float('nan'))

            #     # L_SISNR: SI-SNR loss
            #     # not used for stft model
            #     g_loss_SISNR = torch.tensor(float('nan'))

            #     # Total loss
            #     g_loss = g_loss_id + self.lambda_cd * g_loss_cd

            elif self.model_type == 'wav':
                x_convtas, x_identic, gen_outputs, code_real = self.G(x_real, emb_org, emb_org)

                # L_recon
                g_loss_id = F.mse_loss(x_real.squeeze(), x_identic.squeeze())

                # loss for generator
                g_loss_gen = F.mse_loss(x_convtas.squeeze(), gen_outputs.squeeze()) # loss for data generated specs
                
                # L_content: Code semantic loss
                code_reconst = self.G(x_identic, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                # set postnet loss to nan (since we found out that postnet makes no difference)
                g_loss_id_psnt = torch.tensor(float('nan')).to(self.device)

                # L_SISNR: SI-SNR loss
                dot = torch.sum(x_identic * x_real, dim=1, keepdim=True)
                s_target_energy = torch.sum(x_real ** 2, dim=1, keepdim=True)
                scaled_target = dot * x_real / s_target_energy
                e_noise = x_identic - scaled_target
                losses = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1))
                losses = (10 * torch.log10(losses))
                g_loss_SISNR = -(losses.mean())

                # Total loss
                g_loss = g_loss_id + self.lambda_SISNR * g_loss_SISNR + g_loss_gen + self.lambda_cd * g_loss_cd
            else: print('Model type not recognized')
                
            self.reset_grad()
            g_loss.backward()

            # is this neccessary?
            #for param in self.G.parameters():
            #    param.grad.data.clamp_(-1,1)
            
            self.g_optimizer.step()

            # Learning rate scheduler step! OBS! 
            if self.lr_scheduler is not None:
                if self.lr_scheduler == 'Cosine': 
                    self.lr_scheduler.step()
                    lr = self.lr_scheduler.get_last_lr()[0]
                    print('The current convtas learning rate:', lr)
                else: # Plateau
                    self.lr_scheduler.step(g_loss) # the loss should be validation loss not training loss..
                    lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
                    print('The current convtas learning rate:', lr)

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                #print(log)

                # Save model checkpoint.
                self.model_EMA() # loading model with average parameters
                state = {
                    'epoch': i+1,
                    'state_dict': self.G.state_dict(), # OBS! this is for averaged weights
                    'optimizer': self.g_optimizer.state_dict(),
                    'loss': loss
                }
    
                if self.file_exists:
                    save_name = 'chkpnt_'+self.model_type + '_' + self.run_name+ '_resumed.ckpt'
                else:
                    save_name = 'chkpnt_'+self.model_type + '_' + self.run_name+ '.ckpt'

                torch.save(state, save_name)
                
                if self.model_type in ('stft', 'spmel'):
                    #log melspec
                    fig, axs = plt.subplots(2, 1, sharex=True)
                    display.specshow(
                        x_real[0].T.detach().cpu().numpy() * 100 - 100,
                        y_axis=("mel" if self.model_type == 'spmel' else "fft"),
                        x_axis="time",
                        fmin=90,
                        fmax=7_600, 
                        sr=16_000,
                        ax=axs[0],
                    )
                    axs[0].set(title="Original spectrogram")
                    axs[0].label_outer()

                    # if self.model_type == 'spmel':
                    #     x_identic_plot = (x_identic_psnt[0].T.detach().cpu().numpy() * 100 - 100).squeeze()
                    # else:
                    #     x_identic_plot = (x_identic[0].T.detach().cpu().numpy() * 100 - 100).squeeze()

                    img = display.specshow(
                        x_identic_psnt[0].T.detach().cpu().numpy() * 100 - 100,
                        y_axis=("mel" if self.model_type == 'spmel' else "fft"),
                        x_axis="time",
                        fmin=90,
                        fmax=7_600,
                        sr=16_000,
                        ax=axs[1],
                    )
                    axs[1].set(title="Converted spectrogram")
                    #fig.suptitle(f"{'git money git gud'}") #self.CHECKPOINT_DIR / Path(subject[0]).stem
                    fig.colorbar(img, ax=axs)
                    wandb.log({"Train spectrograms": wandb.Image(fig)}, step=i)
                    plt.close()
                else:
                    mel_spec_real = plot_mel(x_real[0].detach().cpu().numpy())
                    mel_spec_identic = plot_mel(x_identic[0].detach().cpu().numpy())
                                        #log melspec
                    fig, axs = plt.subplots(2, 1, sharex=True)
                    display.specshow(
                        mel_spec_real,
                        y_axis=("mel"),
                        x_axis="time",
                        fmin=90,
                        fmax=7_600, 
                        sr=16_000,
                        ax=axs[0],
                    )
                    axs[0].set(title="Original spectrogram")
                    axs[0].label_outer()

                    img = display.specshow(
                        mel_spec_identic,
                        y_axis=("mel" if self.model_type == 'spmel' else "fft"),
                        x_axis="time",
                        fmin=90,
                        fmax=7_600,
                        sr=16_000,
                        ax=axs[1],
                    )
                    axs[1].set(title="Converted spectrogram")
                    #fig.suptitle(f"{'git money git gud'}") #self.CHECKPOINT_DIR / Path(subject[0]).stem
                    fig.colorbar(img, ax=axs)
                    wandb.log({"Train spectrograms": wandb.Image(fig)}, step=i+1)
                    plt.close()

                # For weights and biases.
                wandb.log({"i": i,
                        "lr": lr,
                        "g_loss": g_loss.item(),
                        "g_loss_id": g_loss_id.item(), # L_recon
                        "g_loss_id_psnt": g_loss_id_psnt.item(), # L_recon0
                        "g_loss_cd": g_loss_cd.item(), # L_content
                        "g_loss_SISNR": g_loss_SISNR.item()}) # L_SISNR

    
    

    