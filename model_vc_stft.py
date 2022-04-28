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

    def forward(self, x, c_org, c_trg):
                
        codes = self.model.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        stft_outputs = self.model.decoder(encoder_outputs)
        
        return stft_outputs, torch.cat(codes, dim=-1)
