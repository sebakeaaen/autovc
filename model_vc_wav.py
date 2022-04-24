import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# implement ConvTasEncoder and ConvTasDecoder modules like this (PRelu not Dilation)
# https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet

class ConvTasNetEncoder(nn.Module):
    def __init__(self, enc_dim=80, sr=16000, kernel_size=3, depth=1):
        super(ConvTasNetEncoder, self).__init__()

        #self.kernel_size = int(sr*kernel_size/1000) # is this neccessary?
        stride = kernel_size // 2
        padding = kernel_size // 2
        
        # input encoder
        if depth == 1:
            self.encoder = nn.Sequential(
                nn.Conv1d(128, 1, kernel_size, stride, padding),
                nn.PReLU())
        elif depth == 3:
            self.encoder = nn.Sequential(
                nn.Conv1d(128, enc_dim, kernel_size, stride, padding),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, 1, kernel_size=3, stride=1, padding=1),
                nn.PReLU())
        elif depth == 5:
            self.encoder = nn.Sequential(
                nn.Conv1d(128, enc_dim, kernel_size, stride, padding),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv1d(enc_dim, 128, kernel_size=3, stride=1, padding=1),
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
            self.decoder = nn.ConvTranspose1d(enc_dim, 128, kernel_size, stride, padding, bias=False)
        elif depth == 3:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, 128, kernel_size, stride, padding, bias=False))
        elif depth == 5:
            self.encoder = nn.Sequential(
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, enc_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ConvTranspose1d(enc_dim, 128, kernel_size, stride, padding, bias=False))
        else: print('Model not defined for this depth')

    def forward(self, x):
        return self.decoder(x)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
            ConvNorm(80+dim_emb if i == 0 else 512,
                        512,
                        kernel_size=5, stride=1,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
            nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, model_type):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class GeneratorWav(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, depth):
        super(GeneratorWav, self).__init__()
        
        self.tasEncoder = ConvTasNetEncoder(depth)
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        #self.postnet = Postnet()
        self.tasDecoder = ConvTasNetDecoder(depth)

    def forward(self, x, c_org, c_trg):

        # pass trough conv tas encoder
        x = self.tasEncoder(x)
        
        # pass through AutoVC model 
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        #mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        #mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)

        # pass through conv tas decoder
        output = self.tasDecoder(mel_outputs)
        
        #mel_outputs = mel_outputs.unsqueeze(1)
        #mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return output, torch.cat(codes, dim=-1)