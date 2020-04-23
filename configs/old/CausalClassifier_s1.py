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
data_train_opt['dataset_name'] = 'causal_dsprite_shape2_scale5_imgs_for_classifier'
data_train_opt['split'] = 'train'
data_train_opt['subset_classes'] = [53, 39, 31,  9,  3, 30,  2, 36,  1, 52]
data_train_opt['img_size'] = 256

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['dataset_name'] = 'causal_dsprite_shape2_scale5_imgs_for_classifier'
data_test_opt['split'] = 'test'
data_test_opt['subset_classes'] = [53, 39, 31,  9,  3, 30,  2, 36,  1, 52]
data_test_opt['img_size'] = 256

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt

model_opt = {
    'model': 'Classifier', # class name
    'filename': 'classifier',
    'num_workers': 4,

    'input_dim': 256*256*1,
    'input_channels': 1,
    'weight_init': 'normal_init',
    'image_size': 256,
    'n_classes': 64,
    
    'batch_size': batch_size,
    'snapshot': 50,
    'console_print': 1,
    
    'epochs': 200,
    'lr_schedule': [(0, 1e-03), (50, 1e-04), (150, 1e-05)],
    'optim_type': 'Adam',
    'random_seed': 1201
}

config['model_opt'] = model_opt
config['algorithm_type'] = 'Classifier_Algorithm'
