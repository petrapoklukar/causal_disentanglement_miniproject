#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:11:30 2020

@author: petrapoklukar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
    
class VAE_Conv2D_v2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt['latent_dim']

        self.device = opt['device']
        self.dropout = opt['dropout']

        #--- Encoder network
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            TempPrintShape('Conv0'),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            TempPrintShape('Conv1'),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            TempPrintShape('Conv2'),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            TempPrintShape('Conv3'),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            TempPrintShape('Conv4'),
            nn.ReLU(True),
            nn.Conv2d(128, 2*self.latent_dim, 1))
        
        self.enc_mean = lambda tensor: tensor[:, :self.latent_dim]
        self.enc_logvar = lambda tensor: tensor[:, self.latent_dim:]

        #--- Decoder network
        self.dec_conv = nn.Sequential(
            nn.Conv2d(self.latent_dim, 128, 1),
            TempPrintShape('Dec Conv0'),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            TempPrintShape('Dec TransConv0'),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            TempPrintShape('Dec TransConv1'),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            TempPrintShape('Dec TransConv2'),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            TempPrintShape('Dec TransConv3'),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            TempPrintShape('Dec TransConv4'))
    

        #--- Weight init
        self.weight_init()

    def weight_init(self):
        """
        Weight initialiser.
        """
        initializer = globals()[self.opt['weight_init']]

        for block in self._modules:
            b = self._modules[block]
            if isinstance(b, nn.Sequential):
                for m in b:
                    initializer(m)
            else:
                initializer(b)

    def encoder(self, x):
        """
        Encoder forward step. Returns mean and log variance.
        """
        # Input (batch_size, Channels=1, Width=28, Height=28)
        x = self.enc_conv(x) # (batch_size, lin_before_latent_dim)
        mean = self.enc_mean(x) # (batch_size, latent_dim)
        logvar = self.enc_logvar(x) # (batch_size, latent_dim)
        return mean, logvar

    def decoder(self, z):
        """
        Decoder forward step. Returns mean. Variance is fixed to 1.
        """
        x1 = self.dec_conv(z)
        return x1


    def sample(self, mean, logvar, sample=False):
        """
        Samples z from the given mean and logvar.
        """
        if self.training or sample:
            std = torch.exp(0.5*logvar)
            eps = torch.empty(std.size(), device=self.device).normal_()
            return eps.mul(std).add(mean)
        else:
            return mean

    def forward(self, x, sample_latent=False, latent_code=False):
        latent_mean, latent_logvar = self.encoder(x)
        z = self.sample(latent_mean, latent_logvar, sample=sample_latent)

        if latent_code:
            return z.squeeze()
        else:
            out_mean = self.decoder(z)
            return out_mean, latent_mean, latent_logvar
        
        
class TempPrintShape(nn.Module):
    def __init__(self, message):
        super(TempPrintShape, self).__init__()
        self.message = message

    def forward(self, feat):
        # print(self.message, feat.shape)
        return feat

class ConvToLin(nn.Module):
    def __init__(self):
        super(ConvToLin, self).__init__()

    def forward(self, feat):
        batch, channels, width, height = feat.shape
        feat = feat.view((batch, channels * width * height))
        return feat
    
class LinToConv(nn.Module):
    def __init__(self, input_dim, n_channels):
        super(LinToConv, self).__init__()
        self.n_channels = n_channels
        self.width = int(np.sqrt((input_dim / n_channels)))

    def forward(self, feat):
        # feat = feat.view((feat.shape[0], self.n_channels, self.width, self.width))
        return feat
    
# 2 versions of weight initialisation
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.Parameter)):
        m.data.fill_(0)
        print('Param_init')


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_out_transpose2D(Hi, stride, padding, kernel_size, output_padding, 
                        dilation=1):
    return (Hi-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
    
if __name__ == '__main___':
    size = 64
    opt = {
            'device': 'cpu',
            'input_channels': 1,
            'latent_dim': 2,
            'out_activation': 'sigmoid',
            'dropout': 0.2,
            'weight_init': 'normal_init',
            'fc_dim': 128,
            'enc_kernel_list': [4, 4, 4, 4],
            'enc_channels': [1, 32, 32, 64, 64],
            'dec_kernel_list': [5, 5, 5, 5],
            'dec_channels': [64, 32, 32, 1],
            'image_size': 64,
            'learn_dec_logvar': True
            }

    net = VAE_Conv2D_v2(opt)
    x = torch.autograd.Variable(torch.FloatTensor(5, opt['input_channels'], size, size).uniform_(-1,1))
    latent_mean, latent_logvar = net.encoder(x)
    z = net.sample(latent_mean, latent_logvar)
    print('\n *- Dimension test')
    out = net(x)
    print('    Input: ', x.shape)
    print('    Latent:', out[-1].shape)
    print('    Output:',out[0].shape)
    print(' *---')