import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model_vc_mel import Encoder, Decoder

# implement ConvTasEncoder and ConvTasDecoder modules like this (PRelu not Dilation)
# https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet

class Conv1DBlock(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 80, conv_channel = 256, kernel_size = 3, padding = 1):
        super(Conv1DBlock, self).__init__()

        self.convBlock = nn.Sequential(
                        nn.Conv1d(in_channel, conv_channel, kernel_size = 1, padding = 0),
                        nn.PReLU(),
                        nn.Conv1d(conv_channel, conv_channel, kernel_size = kernel_size, padding = padding, bias = True),
                        nn.PReLU(),
                        nn.Conv1d(conv_channel, out_channel, kernel_size = 1, padding = 0),
        )

    def forward(self, x):
        return self.convBlock(x)

class ConvTasNetEncoder(nn.Module):
    def __init__(self, enc_dim=80, sr=16000, kernel_size=3, depth=1):
        super(ConvTasNetEncoder, self).__init__()

        #self.kernel_size = int(sr*kernel_size/1000) # is this neccessary?
        stride = kernel_size // 2
        padding = kernel_size // 2
        
        # input encoder
        if depth == 1:
            self.encoder = nn.Sequential(nn.Conv1d(1, enc_dim, kernel_size, stride, padding),
                nn.PReLU())
        elif depth == 3:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, enc_dim, kernel_size, stride, padding),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU())
        elif depth == 5:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, enc_dim, kernel_size, stride, padding),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU())
        else: print('Model not defined for this depth')
    
    def forward(self, x):
        return self.encoder(x)


class ConvTasNetDecoder(nn.Module):
    def __init__(self, enc_dim=80, sr=16000, kernel_size=3, depth=1):
        super(ConvTasNetDecoder, self).__init__()
        
        #self.kernel_size = int(sr*kernel_size/1000) # is this neccessary?
        stride = kernel_size // 2
        padding = kernel_size // 2
        
        # output decoder
        if depth == 1:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(80, 1, kernel_size=3, stride=1, padding=1, bias=False))
        elif depth == 3:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, 1, kernel_size, stride, padding, bias=False))
        elif depth == 5:
            self.encoder = nn.Sequential(
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, 1, kernel_size, stride, padding, bias=False))
        else: print('Model not defined for this depth')

    def forward(self, x):
        return self.decoder(x) 

class GeneratorWav(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, depth):
        super(GeneratorWav, self).__init__()
        
        self.tasEncoder = Conv1DBlock()
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.tasDecoder = ConvTasNetDecoder(depth)

    def forward(self, x, c_org, c_trg):
        x = x.permute(0,2,1)

        # pass trough conv tas encoder
        x = self.tasEncoder(x)

        x = x.permute(0,2,1)

        x_convTas = x.clone()

        # pass through AutoVC model 
        codes = self.encoder(x, c_org)

        if c_trg is None:
            return torch.cat(codes, dim=-1)
            
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        gen_outputs = self.decoder(encoder_outputs)

        # pass through conv tas decoder
        output = self.tasDecoder(gen_outputs.permute(0,2,1)) 
        
        return x_convTas, output.permute(0,2,1), gen_outputs, torch.cat(codes, dim=-1)