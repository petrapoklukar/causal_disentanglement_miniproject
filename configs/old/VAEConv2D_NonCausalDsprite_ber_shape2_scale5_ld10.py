#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:19:38 2020

@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'noncausal_dsprite_shape2_scale5_imgs'
data_train_opt['split'] = 'train'
data_train_opt['img_size'] = 64

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['dataset_name'] = 'noncausal_dsprite_shape2_scale5_imgs'
data_test_opt['split'] = 'test'
data_test_opt['img_size'] = 64

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

vae_opt = {
    'model': 'VAE_Conv2D', # class name
    'filename': 'vae',
    'num_workers': 4,

    'loss_fn': 'fixed decoder variance', # 'learnable full gaussian',
    'learn_dec_logvar': False,
    'decoder_param': 'bernouli',

    'input_dim': 64*64*1,
    'image_size': 64,
    'input_channels': 1,
    'latent_dim': 10,
    'out_activation': 'sigmoid',
    'dropout': 0.3,
    'weight_init': 'normal_init',
    
    'fc_dim': 128,
    'enc_kernel_list': [4, 4, 4, 4],
    'enc_channels': [1, 32, 32, 64, 64],
    'dec_kernel_list': [5, 5, 5, 5],
    'dec_channels': [64, 32, 32, 1],
    'image_size': 64,

    'batch_size': batch_size,
    'snapshot': 3,
    'console_print': 1,
    'beta_warmup': 50,
    'beta_min': 0,
    'beta_max': 6,
    'beta_steps': 100,
    'kl_anneal': True,
    
    'epochs': 200,
    'lr_schedule': [(0, 1e-03), (10, 1e-04), (100, 1e-05)],
    'optim_type': 'Adam',
    'random_seed': 1201
}

config['vae_opt'] = vae_opt
config['algorithm_type'] = 'VAE_Algorithm'
