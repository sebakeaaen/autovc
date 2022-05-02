import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model_vc_mel import Encoder, Decoder

# implement ConvTasEncoder and ConvTasDecoder modules like this (PRelu not Dilation)
# https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet

class ConvTasNetEncoder(nn.Module):
    def __init__(self, in_channel=1, conv_channel=80, depth=1):
        super(ConvTasNetEncoder, self).__init__()

        self.conv1x1 = nn.Conv1d(in_channel, conv_channel, kernel_size = 16, stride = 16//2, padding = 0),
        
        convolutions = []
        for i in range(depth):
            conv_layer = nn.Sequential(
                nn.Conv1d(conv_channel, conv_channel, kernel_size = 3, stride=1, padding = 1),
                nn.PReLU())
            convolutions.append(conv_layer)
        self.convD = nn.ModuleList(convolutions)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.convD(x)
        return x


class ConvTasNetDecoder(nn.Module):
    def __init__(self, conv_channel=80, out_channel=1, depth=1):
        super(ConvTasNetDecoder, self).__init__()
        
        convolutions = []
        for i in range(depth):
            conv_layer = nn.Sequential(
                nn.ConvTranspose1d(conv_channel, conv_channel, kernel_size = 3, stride=1, padding = 1),
                nn.PReLU())
            convolutions.append(conv_layer)
        self.convTD = nn.ModuleList(convolutions)

        self.convT1x1 = nn.Conv1d(conv_channel, out_channel, kernel_size = 16, stride = 16//2, padding = 0),
        
    def forward(self, x):
        x = self.convTD(x)
        x = self.convT1x1(x)
        return x

class GeneratorWav(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, depth):
        super(GeneratorWav, self).__init__()
        
        self.tasEncoder = ConvTasNetEncoder(depth)
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.tasDecoder = ConvTasNetDecoder(depth)

    def forward(self, x, c_org, c_trg):
        x = x.permute(0,2,1)

        # pass through conv tas encoder
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