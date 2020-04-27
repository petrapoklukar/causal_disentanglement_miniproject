#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:52:15 2020

@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'causal_dsprite_shape2_scale5_imgs'
data_train_opt['split'] = 'test'
data_train_opt['img_size'] = 64

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['dataset_name'] = 'causal_dsprite_shape2_scale5_imgs'
data_test_opt['split'] = 'test'
data_test_opt['img_size'] = 64

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

vae_opt = {
    'model': 'VAE_Conv2D_v2', # class name
    'filename': 'vae',
    'num_workers': 4,

    'input_dim': 64*64*1,
    'image_size': 64,
    'input_channels': 1,
    'latent_dim': 10,
    'dropout': 0.3,
    'weight_init': 'normal_init',
    'image_size': 64,

    'batch_size': batch_size,
    'snapshot': 500,
    'console_print': 1,
    'kl_anneal': False,
    
    'epochs': 2000,
    'lr_schedule': [(0, 1e-04)],
    'optim_type': 'Adam',
    'random_seed': 1201
}

config['vae_opt'] = vae_opt
config['algorithm_type'] = 'VAE_Algorithm_v2'
