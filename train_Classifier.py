#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:02:25 2020

@author: petrapoklukar
"""


from __future__ import print_function
import argparse
import os
from importlib.machinery import SourceFileLoader
from algorithms import Classifier_Algorithm as alg
from dataloader import CausalDataset

parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, default='', 
#                     help='config file with parameters of the vae model')
# parser.add_argument('--train', type=int, default=1, 
#                     help='trains the model with the given config')
# parser.add_argument('--chpnt_path', type=str, default='', 
#                     help='path to the checkpoint')
# parser.add_argument('--num_workers', type=int, default=0,      
#                     help='number of data loading workers')
# parser.add_argument('--cuda' , type=bool, default=False, help='enables cuda')
args_opt = parser.parse_args()

# # # # Laptop TESTING
args_opt.config = 'CausalClassifier'
args_opt.train = 1
args_opt.chpnt_path = ''#models/VAE_CausalData_ld2/vae_lastCheckpoint.pth'#'
args_opt.num_workers = 0
args_opt.cuda = None



# Load VAE config file
config_file = os.path.join('.', 'configs', args_opt.config + '.py')
directory = os.path.join('.', 'models', args_opt.config)
if (not os.path.isdir(directory)):
    os.makedirs(directory)

print(' *- Training:')
print('    - Classifier: {0}'.format(args_opt.config))

model_config = SourceFileLoader(args_opt.config, config_file).load_module().config 
model_config['exp_name'] = args_opt.config
model_config['model_opt']['exp_dir'] = directory # the place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.config, config_file))
print(' *- Files will be stored on %s' % (directory))

# Initialise VAE model
model_algorithm = getattr(alg, model_config['algorithm_type'])(model_config['model_opt'])
print(' *- Loaded {0}'.format(model_config['algorithm_type']))

data_train_opt = model_config['data_train_opt']
train_dataset = CausalDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    subset_classes=data_train_opt['subset_classes'])

data_test_opt = model_config['data_test_opt']
test_dataset = CausalDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    subset_classes=data_test_opt['subset_classes'])
assert(test_dataset.dataset_name == train_dataset.dataset_name)
assert(train_dataset.split == 'train')
assert(test_dataset.split == 'test')

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers    
else:
    num_workers = config_file['model_opt']['num_workers']

if args_opt.train:
    model_algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)

