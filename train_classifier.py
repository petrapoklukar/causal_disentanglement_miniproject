#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:27:30 2020

@author: petrapoklukar
"""

import argparse
import os
from importlib.machinery import SourceFileLoader
from algorithms import Classifier_Algorithm as alg
from dataloader import CausalDataset

parser = argparse.ArgumentParser()
parser.add_argument('--classifier_config', type=str, required=True, default='', 
                    help='config file with parameters of the classifier model')
parser.add_argument('--vae_config', type=str, required=True, default='', 
                    help='config file with parameters of the vae model')
parser.add_argument('--generate_data', type=int, default=1, 
                    help='generates data for the classifier using the given vae')
parser.add_argument('--train', type=int, default=1, 
                    help='trains the model with the given config')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--cuda' , type=int, default=1, help='enables cuda')
args_opt = parser.parse_args()


# # # # # Laptop TESTING
# args_opt.classifier_config = 'NonCausalClassifier_ld2'
# args_opt.vae_config = 'VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld2'
# args_opt.generate_data = 1
# args_opt.train = 1
# args_opt.chpnt_path = ''#models/VAE_CausalData_ld2/vae_lastCheckpoint.pth'#'
# args_opt.num_workers = 0
# args_opt.cuda = 0

# Load classifier config file
config_file = os.path.join('.', 'configs', args_opt.classifier_config + '.py')
directory = os.path.join('.', 'models', args_opt.classifier_config)
if (not os.path.isdir(directory)):
    os.makedirs(directory)


model_config = SourceFileLoader(args_opt.classifier_config, config_file).load_module().config 
model_config['exp_name'] = args_opt.classifier_config
model_config['model_opt']['exp_dir'] = directory # the place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.classifier_config, config_file))
print(' *- Files will be stored on %s \n' % (directory))

if args_opt.generate_data:
    import torch
    from architectures.VAE_Conv2D_v2 import VAE_Conv2D_v2 as vae_m
    import numpy as np
    import causal_utils as caus_utils
    import random
    import pickle

    print(' *- Generating data:')
    print('    - Classifier: {0}'.format(args_opt.classifier_config))
    print('    - VAE: {0}'.format(args_opt.vae_config))
    
    # Load VAE config file
    vae_config_file = os.path.join('.', 'configs', args_opt.vae_config + '.py')
    vae_directory = os.path.join('.', 'models', args_opt.vae_config)
    vae_model_config = SourceFileLoader(args_opt.vae_config, vae_config_file).load_module().config 
    
    data_config = model_config['data_gen_opt']
    path_to_vae_model = data_config['path_to_vae_model'].format(args_opt.vae_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model_config['vae_opt']['device'] = device

    try:    
        if data_config['load_checkpoint']:
            checkpoint = torch.load(path_to_vae_model, map_location=device)
            trained_dict = checkpoint['model_state_dict']
            print('    - Loaded checkpoint: {0}'.format(path_to_vae_model))
        else:
            trained_dict = torch.load(path_to_vae_model, map_location=device)
        vae_model = vae_m(vae_model_config['vae_opt']).to(device)
        vae_model.load_state_dict(trained_dict)
        vae_model.eval()
    except:
        raise NotImplementedError('Could not load'.format(path_to_vae_model))
    
    assert(not vae_model.training)
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    imgs = dataset_zip['imgs']
    img_size = vae_model_config['vae_opt']['image_size']

    d_sprite_idx, X_true_data, labels = caus_utils.calc_dsprite_idxs(
        num_samples=data_config['num_samples'], seed=data_config['seed'], 
        constant_factor=data_config['constant_factor'], 
        causal=data_config['causal'], color=0, shape=2, scale=5)
    D_data = caus_utils.make_dataset_d_sprite(
        d_sprite_dataset=imgs, dsprite_idx=d_sprite_idx, img_size=img_size)
    zipped_list = []
    
    for ind in range(len(D_data)):
        tensor_image = torch.from_numpy(D_data[ind]).reshape(
            1, 1, img_size, img_size).float().to(device)
        tensor_label = torch.tensor(labels[ind]).long()
        image_decoded = vae_model(tensor_image)[0].cpu().detach().float().squeeze(0)
        zipped_list.append((image_decoded, tensor_label))
            
    del D_data   
    del vae_model
    
    print(' Label min/max:', min(labels), max(labels))
    random.seed(2610)
    random.shuffle(zipped_list)

    prefix = 'non' if not data_config['causal'] else ''
    filename = '{0}causal_dsprite_shape2_scale5_imgs_for_classifier_ld{1}.pkl'.format(
        prefix, vae_model_config['vae_opt']['latent_dim'])
    
    splitratio = int(len(zipped_list) * 0.15)
    train_data = zipped_list[splitratio:]
    test_data = zipped_list[:splitratio]
    
    print(' Train and test split lengths:', len(train_data), len(test_data))
    print(' An element is of type ', type(train_data[0]))
    print('    whereas image and label ', train_data[0][0].dtype, train_data[0][1].dtype)

    with open('datasets/train_' + filename, 'wb') as f:
        pickle.dump(train_data, f)
    with open('datasets/test_' + filename, 'wb') as f:
        pickle.dump(test_data, f)
    
    del train_data
    del test_data
    print(' Data generation completed.\n')

print(' *- Training models:')
print('    - Classifier: {0}'.format(args_opt.classifier_config))

# Initialise classifier model
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