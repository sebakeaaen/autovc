from functools import lru_cache
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
import numpy as np
import os


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        self.main_dir = config.main_dir

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.lr = config.lr
        self.lr_scheduler = config.lr_scheduler
        self.run_name = config.run_name
        self.depth = config.depth

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # models
        self.model_type = config.model_type
        self.speaker_embed = config.speaker_embed

        self.config = config

        #wandb setup
        with open('wandb.token', 'r') as file:
            api_key = file.readline()
            wandb.login(key=api_key)

        self.path = 'chkpnt_' + self.model_type + '_' + self.run_name + '.ckpt'
        self.file_exists = os.path.exists(self.path)

        if self.file_exists:
            wandb.init(project="DNS autovc", entity="macaroni", resume=True, id=self.run_name)
        else:
            wandb.init(project="DNS autovc", entity="macaroni", reinit=True, name=self.run_name)

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
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

        # Set up weights and biases config
        wandb.config.update(config, allow_val_change=True)


    def build_model(self):
        
        if self.model_type == 'spmel':
            self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        elif self.model_type == 'stft':
            self.G = GeneratorSTFT(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)  
        elif self.model_type == 'wav':
            self.G = GeneratorWav(self.dim_neck, self.dim_emb, self.dim_pre, self.freq, self.depth)
        else: print('Model type not recognized')

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
            print('No learning rate scheduler used')
            self.lr_scheduler = None

        if self.file_exists:
            print('Resuming from checkpoint.')
            checkpoint=torch.load(self.path, map_location=self.device)
            self.g_optimizer.load_state_dict(checkpoint['optimizer'])
            self.G.load_state_dict(checkpoint['state_dict'])
            self.epoch = checkpoint['epoch']
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        lr = self.lr

        num_iter = 0
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            if self.file_exists:
                num_iter = self.epoch

            num_iter += 1

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
            
            self.G.to(self.device)
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)

            # L_recon
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   

            # L_content: Code semantic loss
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            if self.model_type == 'spmel':
                # L_recon0
                g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   

                # L_SISNR: SI-SNR loss
                # not used for mel model
                g_loss_SISNR = torch.tensor(float('nan'))

                # Total loss
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            elif self.model_type == 'stft':
                # L_recon0
                # set postnet loss to nan (since we found out that postnet makes no difference)
                g_loss_id_psnt = torch.tensor(float('nan'))

                # L_SISNR: SI-SNR loss
                # not used for stft model
                g_loss_SISNR = torch.tensor(float('nan'))

                # Total loss
                g_loss = g_loss_id + self.lambda_cd * g_loss_cd

            elif self.model_type == 'wav':
                # L_recon
                # set postnet loss to nan (since we found out that postnet makes no difference)
                g_loss_id_psnt = torch.tensor(float('nan'))

                # L_SISNR: SI-SNR loss
                #g_loss_SISNR = 

                # Total loss
                g_loss = g_loss_id + self.lambda_cd * g_loss_cd + self.lambda_SISNR * g_loss_SISNR
            else: print('Model type not recognized')
                

            self.reset_grad()
            g_loss.backward()
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
            if (num_iter) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, num_iter, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

                # Save model checkpoint.
                state = {
                    'epoch': num_iter,
                    'state_dict': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                    'loss': loss
                }
                if self.file_exists:
                    save_name = 'chkpnt_'+self.model_type + '_' + self.run_name+ '_resumed.ckpt'
                else:
                    save_name = 'chkpnt_'+self.model_type + '_' + self.run_name+ '.ckpt'
                torch.save(state, save_name)
                
                #log melspec
                fig, axs = plt.subplots(2, 1, sharex=True)
                print((x_real[0].T.detach().cpu().numpy() * 100 - 100).shape)
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

                x_identic_plot = (x_identic[0].T.detach().cpu().numpy() * 100 - 100).squeeze()
                print(x_identic_plot.shape)
                img = display.specshow(
                    x_identic_plot,
                    y_axis=("mel" if self.model_type == 'spmel' else "fft"),
                    x_axis="time",
                    fmin=90,
                    fmax=7_600,
                    sr=16_000,
                    ax=axs[1],
                )
                axs[1].set(title="Converted spectrogram")
                fig.suptitle(f"{'git money git gud'}") #self.CHECKPOINT_DIR / Path(subject[0]).stem
                fig.colorbar(img, ax=axs)
                wandb.log({"Train spectrograms": wandb.Image(fig)}, step=i)
                plt.close()
                
            # For weights and biases.
            wandb.log({"epoch": num_iter,
                    "lr": lr,
                    "L_recon": g_loss_id.item(),
                    "L_recon0": g_loss_id_psnt.item(),
                    "L_content": g_loss_cd.item(),
                    "L_SISNR": g_loss_SISNR.item()})

            wandb.watch(self.G, log = None)
                

    
    

    