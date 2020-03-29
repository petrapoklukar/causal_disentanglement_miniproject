#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:41:40 2020

@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'causal_data'
data_train_opt['split'] = 'train'
data_train_opt['img_size'] = 256

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['dataset_name'] = 'causal_data'
data_test_opt['split'] = 'test'
data_test_opt['img_size'] = 256

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

vae_opt = {
    'model': 'VAE_TinyResNet', # class name
    'filename': 'vae',
    'num_workers': 4,

    'loss_fn': 'fixed decoder variance', # 'learnable full gaussian',
    'learn_dec_logvar': False,
    'input_dim': 256*256*3,
    'input_channels': 3,
    'latent_dim': 10,
    'out_activation': 'sigmoid',
    'dropout': 0.3,
    'weight_init': 'normal_init',

    'conv1_out_channels': 32,
    'latent_conv1_out_channels': 128,
    'kernel_size': 3,
    'num_scale_blocks': 2,
    'block_per_scale': 1,
    'depth_per_block': 2,
    'fc_dim': 512,
    'image_size': 256,

    'batch_size': batch_size,
    'snapshot': 50,
    'console_print': 1,
    'beta_min': 1,
    'beta_max': 1,
    'beta_steps': 1,
    'kl_anneal': False,
    
    'epochs': 200,
    'lr_schedule': [(0, 1e-04), (20, 1e-05), (100, 1e-6)],
    'optim_type': 'Adam',
    'random_seed': 1201
}

config['vae_opt'] = vae_opt
config['algorithm_type'] = 'VAE_Algorithm'
