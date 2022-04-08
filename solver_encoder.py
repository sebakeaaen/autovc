from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import wandb
import matplotlib.pyplot as plt
from librosa import display


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.lr = config.learning_rate
        self.lr_scheduler = config.lr_scheduler
        self.run_name = config.run_name

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
        wandb.config.update(config)


    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq, self.speaker_embed)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr)

        if self.lr_scheduler == 'Cosine': 
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=10000, eta_min=0)
        elif self.lr_scheduler == 'Plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
        else:
            print('No learning rate scheduler used')
            self.lr_scheduler = None
            
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        lr = self.lr
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

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
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Learning rate scheduler step
            if self.lr_scheduler is not None:
                if self.lr_scheduler == 'Cosine': 
                    self.lr_scheduler.step()
                    lr = self.lr_scheduler.get_last_lr()[0]
                    print('The current learning rate:', lr)
                else: # Plateau
                    self.lr_scheduler.step(g_loss) # the loss should be validation loss not training loss..
                    lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
                    print('The current learning rate:', lr)

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
                print(log)

                # Save model checkpoint.
                state = {
                    'epoch': i+1,
                    'state_dict': self.G.state_dict(),
                }
                save_name = 'chkpnt_'+self.model_type + '_' + self.run_name+ '.ckpt'
                torch.save(state, save_name)
                
                #log melspec
                fig, axs = plt.subplots(2, 1, sharex=True)
                print((x_real[0].T.detach().cpu().numpy() * 100 - 100).shape)
                display.specshow(
                    x_real[0].T.detach().cpu().numpy() * 100 - 100,
                    y_axis="mel",
                    x_axis="time",
                    fmin=90,
                    fmax=7_600,
                    sr=16_000,
                    ax=axs[0],
                )
                axs[0].set(title="Original Mel spectrogram")
                axs[0].label_outer()
                print((x_identic_psnt[0].T.detach().cpu().numpy() * 100 - 100).shape)
                img = display.specshow(
                    x_identic_psnt[0].T.detach().cpu().numpy() * 100 - 100,
                    y_axis="mel",
                    x_axis="time",
                    fmin=90,
                    fmax=7_600,
                    sr=16_000,
                    ax=axs[1],
                )
                axs[1].set(title="Converted Mel spectrogram")
                fig.suptitle(f"{'git money git gud'}") #self.CHECKPOINT_DIR / Path(subject[0]).stem
                fig.colorbar(img, ax=axs)
                wandb.log({"Train mel spectrograms": wandb.Image(fig)}, step=i)
                plt.close()
            
            # For weights and biases.
            wandb.log({"epoch": i+1,
                    "lr": lr,
                    "g_loss_id": g_loss_id.item(),
                    "g_loss_id_psnt": g_loss_id_psnt.item(),
                    "g_loss_cd": g_loss_cd.item()})

            wandb.watch(self.G, log = None)
                

    
    

    