#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:58:26 2019

@author: petrapoklukar
"""

from __future__ import print_function
import argparse
import os
from importlib.machinery import SourceFileLoader
from algorithms import VAE_Algorithm as alg
from dataloader import CausalDataset
import prd_score as prd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--exp_vae', type=str, required=True, default='', 
                    help='config file with parameters of the vae model')
parser.add_argument('--train', type=int, default=1, 
                    help='trains the model with the given config')
parser.add_argument('--chpnt_path', type=str, default='', 
                    help='path to the checkpoint')
parser.add_argument('--num_workers', type=int, default=0,      
                    help='number of data loading workers')
parser.add_argument('--compute_prd' , type=int, default=1, 
                    help='computes PRD scores for the model')
parser.add_argument('--cuda' , type=bool, default=False, help='enables cuda')
args_opt = parser.parse_args()

# # # # Laptop TESTING
# args_opt.exp_vae = 'VAE_CausalDsprite_shape2_scale5_ld2'
# args_opt.train = 1
# args_opt.chpnt_path = ''#models/VAE_CausalData_ld2/vae_lastCheckpoint.pth'#'
# args_opt.num_workers = 0
# args_opt.cuda = None
# args_opt.compute_prd = 1


# Load VAE config file
vae_config_file = os.path.join('.', 'configs', args_opt.exp_vae + '.py')
vae_directory = os.path.join('.', 'models', args_opt.exp_vae)
if (not os.path.isdir(vae_directory)):
    os.makedirs(vae_directory)

print(' *- Training:')
print('    - VAE: {0}'.format(args_opt.exp_vae))

vae_config = SourceFileLoader(args_opt.exp_vae, vae_config_file).load_module().config 
vae_config['exp_name'] = args_opt.exp_vae
vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
print(' *- Loading experiment %s from file: %s' % (args_opt.exp_vae, vae_config_file))
print(' *- Generated logs, snapshots, and model files will be stored on %s' % (vae_directory))

# Initialise VAE model
vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
print(' *- Loaded {0}'.format(vae_config['algorithm_type']))

data_train_opt = vae_config['data_train_opt']
train_dataset = CausalDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'])

data_test_opt = vae_config['data_test_opt']
test_dataset = CausalDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'])
assert(test_dataset.dataset_name == train_dataset.dataset_name)
assert(train_dataset.split == 'train')
assert(test_dataset.split == 'test')

if args_opt.num_workers is not None:
    num_workers = args_opt.num_workers    
else:
    num_workers = vae_config_file['vae_opt']['num_workers']

if args_opt.train:
    vae_algorithm.train(train_dataset, test_dataset, num_workers, args_opt.chpnt_path)

# Evaluate the model with precision and recall
if args_opt.compute_prd:
    import torch
    from sklearn.random_projection import GaussianRandomProjection
    
    if not args_opt.train:
        model_path = 'models/{0}/vae_lastCheckpoint.pth'.format(args_opt.exp_vae)
        vae_algorithm.load_checkpoint(model_path, eval=True)
        
    # Compute prd scores
    def sample_prior(vae):
        with torch.no_grad():
            ld = vae.latent_dim
            enc_mean = torch.zeros(ld, device=vae.device)
            enc_std = torch.ones(ld, device=vae.device)
            latent_normal = torch.distributions.normal.Normal(enc_mean, enc_std)
            z_samples = latent_normal.sample((n_samples, ))
            dec_z_samples, _ = vae.decoder(z_samples)
            dec_z_samples = dec_z_samples.detach().cpu().numpy().reshape(n_samples, -1)
            eval_np = transformer.transform(dec_z_samples)
        return eval_np


    n_samples = 500
    chpnt1, chpnt2 = 47, 6
    
    # Fit a random projection on a subset of training data
    rng = np.random.RandomState(42)
    transformer = GaussianRandomProjection(random_state=rng, n_components=1000)
    proj_train_data = train_dataset.get_subset(len(train_dataset), 1000)
    transformer.fit(proj_train_data)

    # Transform test samples and generated samples with it
    ref_np_original = test_dataset.get_subset(len(test_dataset), n_samples)
    ref_np = transformer.transform(ref_np_original)
    
    # Fully trained model
    model_np = sample_prior(vae_algorithm.model)
    
    # Load a chpt for comparisson
    vae_garbage = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    chp1_path = 'models/{0}/vae_checkpoint{1}.pth'.format(args_opt.exp_vae, chpnt1)
    vae_garbage.load_checkpoint(chp1_path, eval=True)
    chpnt1_np = sample_prior(vae_garbage.model)
    
    # Init a new model
    vae_garbage2 = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    chp2_path = 'models/{0}/vae_checkpoint{1}.pth'.format(args_opt.exp_vae, chpnt2)
    vae_garbage2.load_checkpoint(chp2_path, eval=True)
    chpnt2_np = sample_prior(vae_garbage2.model)
        
    # Compute prd
    prd_data_model = prd.compute_prd_from_embedding(model_np, ref_np)
    prd_data_chpnt1 = prd.compute_prd_from_embedding(chpnt1_np, ref_np)
    prd_data_chpnt2 = prd.compute_prd_from_embedding(chpnt2_np, ref_np)
    prd.plot([prd_data_model, prd_data_chpnt1, prd_data_chpnt2], 
             [args_opt.exp_vae, 'chpt' + str(chpnt1), 'chpt' + str(chpnt2)], 
             out_path='models/{0}/prd.png'.format(args_opt.exp_vae))