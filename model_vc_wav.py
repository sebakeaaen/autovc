import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model_vc_mel import Encoder, Decoder, LinearNorm, ConvNorm

# implement ConvTasEncoder and ConvTasDecoder modules like this (PRelu not Dilation)
# https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet

class ConvTasNetEncoder(nn.Module):
    def __init__(self, depth, bias=True):
        super(ConvTasNetEncoder, self).__init__()
        N=512
        L=1024
        S=256

        self.conv1x1 = nn.Conv1d(1, N, kernel_size = L, stride = S, padding = 0, bias=bias)

        convolutions = []
        for i in range(depth):
            conv_layer = nn.Sequential(
                nn.Conv1d(N, N, kernel_size = 3, stride=1, padding = 1, bias=bias),
                nn.PReLU(),
                nn.BatchNorm1d(N))
            convolutions.append(conv_layer)
        self.convD = nn.ModuleList(convolutions)
    
    def forward(self, x):
        x = self.conv1x1(x)
        for conv in self.convD:
            x = conv(x)
        return x


class ConvTasNetDecoder(nn.Module):
    def __init__(self, depth, bias=True):
        super(ConvTasNetDecoder, self).__init__()
        N=512
        L=1024
        S=256

        convolutions = []
        for i in range(depth):
            conv_layer = nn.Sequential(
                nn.ConvTranspose1d(N, N, kernel_size = 3, stride=1, padding = 1, bias=bias),
                nn.PReLU(),
                nn.BatchNorm1d(N))
            convolutions.append(conv_layer)
        self.convTD = nn.ModuleList(convolutions)

        self.convT1x1 = nn.ConvTranspose1d(N, 1, kernel_size = L, stride = S, padding = 0, bias=bias)
        
    def forward(self, x):
        for convT in self.convTD:
            x = convT(x)
        x = self.convT1x1(x)
        return x

class GeneratorWav(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, depth):
        super(GeneratorWav, self).__init__()
        N = 512

        self.tasEncoder = ConvTasNetEncoder(depth)

        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.encoder.convolutions[0][0] = ConvNorm(N+dim_emb, 512, kernel_size = 5, stride = 1, padding = 2)
        self.decoder.linear_projection = LinearNorm(in_dim = 1024, out_dim = N)

        self.tasDecoder = ConvTasNetDecoder(depth)

    def forward(self, x, c_org, c_trg):
        # inputs:
        # x dim: [2, 33536, 1]
        # c_org dim: [2, 256]
        # c_trg dim: [2, 256]

        # pass through conv tas encoder
        x = x.permute(0,2,1) # dim: [2, 1, 33536]
        x = self.tasEncoder(x) # dim: [2, 512, 128]
        x_CTencoder = x.clone() # dim: [2, 128, 512]

        # pass through AutoVC model 
        x = x.permute(0,2,1) # dim: [2, 128, 512]
        codes = self.encoder(x, c_org)

        if c_trg is None:
            return torch.cat(codes, dim=-1)
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)

        x_encoder = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1) # dim: [2, 128, 320]
        x_decoder = self.decoder(x_encoder).permute(0,2,1) # dim: [2, 512, 128]

        # pass through conv tas decoder
        x_identic = self.tasDecoder(x_decoder).permute(0,2,1) # dim: [2, 33536, 1]
        code_real = torch.cat(codes, dim=-1)
        return x_CTencoder, x_identic, x_decoder, code_real