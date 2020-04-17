#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:11:01 2020

@author: petrapoklukar
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvToLin(nn.Module):
    def __init__(self):
        super(ConvToLin, self).__init__()

    def forward(self, feat):
        batch, channels, width, height = feat.shape
        feat = feat.view((batch, channels * width * height))
        return feat


class TempPrintShape(nn.Module):
    def __init__(self, message):
        super(TempPrintShape, self).__init__()
        self.message = message

    def forward(self, feat):
        # print(self.message, feat.shape)
        return feat


class Classifier(nn.Module):
    """Simple classifier."""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_classes = opt['n_classes']
        self.device = opt['device']
        self.input_channels = opt['input_channels']
        
        #--- Encoder network
        self.conv = nn.Sequential(
            TempPrintShape('Input'),
            nn.Conv2d(self.input_channels, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            TempPrintShape('Output of conv0'),
            nn.MaxPool2d(5),
            TempPrintShape('Output of maxpool'),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            TempPrintShape('Output of conv1'),
            nn.MaxPool2d(5),
            TempPrintShape('Output of maxpool'),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            TempPrintShape('Output of conv2'),
            ConvToLin(),
            nn.Linear(6400, 1024),
            nn.ReLU(),
            TempPrintShape('Output of lin'),
            nn.Linear(1024, self.n_classes),
            TempPrintShape('Output'))
        
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

    def forward(self, x):
        return self.conv(x)


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


def create_model(opt):
    return Classifier(opt)


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    size = 256
    opt = {
            'device': 'cpu',
            'input_channels': 1,
            'latent_dim': 2,
            'n_classes': 64,
            'dropout': 0.2,
            'weight_init': 'normal_init',
            'image_size': 256
            }

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(5, opt['input_channels'], size, size).uniform_(-1,1))
    out = net(x)
    
    print('\n * ---')
    print(x.shape)
    print(out.shape)
    print(' * ---')

