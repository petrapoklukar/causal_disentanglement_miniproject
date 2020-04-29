#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:07:20 2020

@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'noncausal_dsprite_shape2_scale5_imgs_for_classifier_ld6'
data_train_opt['split'] = 'train'
data_train_opt['subset_classes'] = []
data_train_opt['img_size'] = 64

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['dataset_name'] = 'noncausal_dsprite_shape2_scale5_imgs_for_classifier_ld6'
data_test_opt['split'] = 'test'
data_test_opt['subset_classes'] = []
data_test_opt['img_size'] = 64

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

data_gen_opt = {
    'path_to_vae_model': 'models/{0}/vae_lastCheckpoint.pth',
    'load_checkpoint': True,
    'num_samples': 20000,
    'seed': 999,
    'constant_factor': [0, 0, 0], 
    'causal': False
}

config['data_gen_opt']  = data_gen_opt

model_opt = {
    'model': 'Classifier', # class name
    'filename': 'classifier',
    'num_workers': 4,

    'input_dim': 64*64*1,
    'input_channels': 1,
    'weight_init': 'normal_init',
    'image_size': 64,
    'n_classes': 8*8*4,
    'latent_dim': 6,
    
    'batch_size': batch_size,
    'snapshot': 10,
    'console_print': 1,
    
    'epochs': 300,
    'min_epochs': 100,
    'max_epochs': 300,
    'lr_schedule': [(0, 1e-03)],
    'optim_type': 'Adam',
    'random_seed': 1201
}

config['model_opt'] = model_opt
config['algorithm_type'] = 'Classifier_Algorithm'
