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
    
class VAE_Conv2D(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.latent_dim = opt['latent_dim']

        self.device = opt['device']
        self.dropout = opt['dropout']
        self.out_activation = opt['out_activation']
        self.learn_dec_logvar = opt['learn_dec_logvar']
        self.decoder_fn = self.decoder_mean_var if opt['learn_dec_logvar'] else self.decoder_mean

        self.fc_dim = opt['fc_dim']        
        self.enc_kernel_list = opt['enc_kernel_list']  
        self.enc_channels = opt['enc_channels']  
        
        self.dec_kernel_list = opt['dec_kernel_list']  
        self.dec_channels = opt['dec_channels']  
        # self.dec_channels = [64, 64, 32, 32, 1] # For TransposeConv2d

        #--- Encoder network
        self.enc_conv = nn.Sequential()
        for ind in range(len(self.enc_channels)-1):
            self.enc_conv.add_module('enc_conv'+str(ind), nn.Conv2d(
                self.enc_channels[ind], self.enc_channels[ind+1], 
                self.enc_kernel_list[ind],
                stride=2, padding=int((self.enc_kernel_list[ind]-1)/2)))
            self.enc_conv.add_module('enc_relu'+str(ind), nn.ReLU())
            if ind % 2 == 1:
                self.enc_conv.add_module('enc_dropout'+str(ind), nn.Dropout(
                    self.dropout))
            self.enc_conv.add_module(
                'out_conv'+str(ind), TempPrintShape('Enc conv'+str(ind)))
        
        self.enc_conv.add_module('convtolin', ConvToLin()) 
        self.enc_conv.add_module('end_lin1', nn.Linear(1024, self.fc_dim))
        self.enc_conv.add_module('out_lin1', TempPrintShape('Enc lin1'))

        self.enc_mean = nn.Linear(self.fc_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(self.fc_dim, self.latent_dim)

        #--- Decoder network
        self.dec_conv = nn.Sequential()
        self.dec_conv.add_module('start_lin1', nn.Linear(self.latent_dim, self.fc_dim))
        self.dec_conv.add_module('out_lin1', TempPrintShape('Dec lin1'))
        self.dec_conv.add_module('start_lin2', nn.Linear(self.fc_dim, 4*4*64))
        self.dec_conv.add_module('out_lin2', TempPrintShape('Dec lin2'))
        self.dec_conv.add_module('convtolin', LinToConv(4*4*64, 64)) 
        self.dec_conv.add_module('out_conv', TempPrintShape('Dec to conv'))
        
        # USING ConvTranspose2d
        # for ind in range(len(self.dec_channels)-2):
        #     self.dec_conv.add_module('dec_conv'+str(ind), nn.ConvTranspose2d(
        #         self.dec_channels[ind], self.dec_channels[ind+1], 
        #         self.dec_kernel_list[ind], stride=2, padding=1, output_padding=0))

        #     self.dec_conv.add_module('out_conv'+str(ind), 
        #                              TempPrintShape('Dec conv'+str(ind)))
        
        for ind in range(len(self.dec_channels)-2):
            self.dec_conv.add_module('dec_up'+str(ind), nn.Upsample(scale_factor=4))
            self.dec_conv.add_module('dout_up'+str(ind), 
                                      TempPrintShape('Dec up'+str(ind)))
            self.dec_conv.add_module('dec_conv'+str(ind), nn.Conv2d(
                self.dec_channels[ind], self.dec_channels[ind+1], 
                self.dec_kernel_list[ind], stride=1, padding=2))
            self.enc_conv.add_module('enc_relu'+str(ind), nn.ReLU())
            if ind % 2 == 0:
                self.dec_conv.add_module('enc_dropout'+str(ind), nn.Dropout(
                    self.dropout))
            self.dec_conv.add_module('dout_conv'+str(ind), 
                                      TempPrintShape('Dec conv'+str(ind)))
            

        # Output mean
        self.dec_mean = nn.Sequential()
        self.dec_mean.add_module('dec_conv3', nn.Conv2d(
                self.dec_channels[-2], self.dec_channels[-1], 
                self.dec_kernel_list[-1], stride=1, padding=2))
        self.dec_mean.add_module('out_conv3', TempPrintShape('Dec conv3'))
        if opt['out_activation'] == 'sigmoid':
            self.dec_mean.add_module('dec_meanact', nn.Sigmoid())

        # Output var as well
        if self.learn_dec_logvar:
            self.dec_logvar = nn.Conv2d(
                self.dec_channels[-2], self.dec_channels[-1], 
                self.dec_kernel_list[-1], stride=1, padding=2)
            print(' *- Learned likelihood variance.')
        print(' *- Last layer activation function: ', self.out_activation)

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
        Decoder forward step. Points to the correct decoder depending on whether
        or not the variance of the likelihood function is learned or not.
        """
        return self.decoder_mean_var(z) if self.learn_dec_logvar else self.decoder_mean(z)

    def decoder_mean(self, z):
        """
        Decoder forward step. Returns mean. Variance is fixed to 1.
        """
        x1 = self.dec_conv(z)
        mean = self.dec_mean(x1)
        logvar = torch.zeros(mean.shape, device=self.device)
        return mean, logvar

    def decoder_mean_var(self, z):
        """
        Decoder forward step. Returns mean and log variance.
        """
        x1 = self.dec_conv(z)
        mean = self.dec_mean(x1)
        logvar = self.dec_logvar(x1)
        return mean, logvar

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
            out_mean, out_logvar = self.decoder_fn(z)
            return out_mean, out_logvar, latent_mean, latent_logvar
        
        
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
        feat = feat.view((feat.shape[0], self.n_channels, self.width, self.width))
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
    
if __name__ == '__main__':
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

    net = VAE_Conv2D(opt)
    x = torch.autograd.Variable(torch.FloatTensor(5, opt['input_channels'], size, size).uniform_(-1,1))
    latent_mean, latent_logvar = net.encoder(x)
    z = net.sample(latent_mean, latent_logvar)
    print('\n *- Dimension test')
    out = net(x)
    print('    Input: ', x.shape)
    print('    Latent:', out[-1].shape)
    print('    Output:',out[0].shape)
    print(' *---')