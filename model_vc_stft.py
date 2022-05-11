import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_vc_mel import LinearNorm, ConvNorm, Generator

class GeneratorSTFT(nn.Module):
    """Generator network for STFT. Based on the pretrained model AutoVC"""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(GeneratorSTFT, self).__init__()
        
        # load pretrained model
        self.model = Generator(dim_neck,dim_emb,dim_pre,freq)

        # modify en- and decoder to STFT sizes
        self.model.encoder.convolutions[0][0] = ConvNorm(513+dim_emb, 512, kernel_size = 5, stride = 1, padding = 2)
        self.model.decoder.linear_projection = LinearNorm(in_dim = 1024, out_dim = 513)

        # modify postnet to STFTs
        self.model.postnet.convolutions[0][0] = ConvNorm(513, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh')
        self.model.postnet.convolutions[4] = nn.Sequential(
                ConvNorm(512, 513,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(513))

    def forward(self, x, c_org, c_trg):
                
        codes = self.model.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        x_identic = self.decoder(encoder_outputs)

        x_identic_psnt = self.postnet(x_identic.transpose(2,1))
        x_identic_psnt = x_identic + x_identic_psnt.transpose(2,1)

        x_identic = x_identic.unsqueeze(1)
        x_identic_psnt = x_identic_psnt.unsqueeze(1)
        code_real = torch.cat(codes, dim=-1)

        return x_identic, x_identic_psnt, code_real
'''
class GeneratorTasNet(nn.Module):
    """Generator network for STFT. Based on the pretrained model AutoVC"""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(GeneratorSTFT, self).__init__()
        
        # load pretrained model
        self.model = Generator(dim_neck,dim_emb,dim_pre,freq)
        checkpoint = torch.load('autovc.ckpt', map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        # freze all weights
        for param in self.model.parameters():
            param.requires_grad = False

        # modify en- and decoder to STFT sizes
        self.model.encoder.convolutions[0][0] = ConvNorm(513+dim_emb, 512, kernel_size = 5, stride = 1, padding = 2, w_init_gain='tanh')
        self.model.decoder.linear_projection = LinearNorm(in_dim = 1024, out_dim = 513)

        # modify the postnet for STFTs
        self.model.postnet = nn.ModuleList()

        self.model.postnet.append(
            nn.Sequential(
                ConvNorm(513, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.model.postnet.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.model.postnet.append(
            nn.Sequential(
                ConvNorm(512, 513, # for mel specs
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(513))
            )

    def forward(self, x, c_org, c_trg):
                
        codes = self.model.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        x_identic = self.model.decoder(encoder_outputs)

        x_identic_psnt = self.model.postnet(x_identic.transpose(2,1))
        x_identic_psnt = x_identic + x_identic_psnt.transpose(2,1)
        
        code_real = torch.cat(codes, dim=-1)

        return x_identic, x_identic_psnt, code_real
        
        # old notation
        #stft_outputs = self.model.decoder(encoder_outputs)
                
        # we dont use the postnet for stfts
        #stft_outputs_postnet = self.model.postnet(stft_outputs.transpose(2,1))
        #stft_outputs_postnet = stft_outputs + stft_outputs_postnet.transpose(2,1)
        
        #return stft_outputs, stft_outputs_postnet, torch.cat(codes, dim=-1)
'''
