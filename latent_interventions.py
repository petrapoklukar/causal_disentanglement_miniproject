#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:20:07 2020

@author: petrapoklukar
"""

from __future__ import print_function
import os
from importlib.machinery import SourceFileLoader
import numpy as np
from architectures.VAE_TinyResNet import VAE_TinyResNet as vae_m
from architectures.Classifier import Classifier as classifier_m
import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import causal_utils as caus_utils

def init_vae(opt):
    """Initialises the VAE model."""
    try:
        instance = vae_m(opt).to(opt['device'])
        trained_dict = torch.load(opt['exp_dir'] + '/vae_model.pt', map_location=opt['device'])
        instance.load_state_dict(trained_dict)
        return instance
    except:
        raise NotImplementedError(
                'Model {0} not recognized'.format(opt['model']))
        
def init_classifier(opt):
    """Initialises the classifier."""
    try:
        instance = classifier_m(opt).to(opt['device'])
        trained_dict = torch.load(opt['exp_dir'] + '/classifier_model.pt', map_location=opt['device'])
        instance.load_state_dict(trained_dict)
        return instance
    except:
        raise NotImplementedError(
                'Model {0} not recognized'.format(opt['model']))
        
        
def get_config(exp_name, key='vae_opt'):
    config_file = os.path.join('.', 'configs', exp_name + '.py')
    directory = os.path.join('.', 'models', exp_name)
    model_config = SourceFileLoader(exp_vae, config_file).load_module().config 
    model_config['exp_name'] = exp_name
    model_config[key]['exp_dir'] = directory 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config['device'] = device
    model_config[key]['device'] = device
    return model_config

            
def sample_latent_codes(ld, n_samples, vae, classifier, device='cpu', random_seed=1234):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    latent_codes_dict = {}
    
    for dim in range(ld):
        fixed_value = torch.empty(1).normal_()
        fixed_dim = fixed_value.expand((1, n_samples))
        
        random_noise = torch.empty((n_samples, ld)).normal_() 
        random_noise[:, dim] = fixed_dim
        
        with torch.no_grad():
            decoded = vae.decoder(random_noise)[0].detach()
            classes = classifier(decoded).detach()
            out = torch.nn.Softmax(dim=1)(classes)
            out_argmax = torch.max(out, dim=1)
        
        latent_codes_dict[str(dim)] = out_argmax[1]
        
    return latent_codes_dict


def load_models(exp_vae, exp_classifier):
    vae_config = get_config(exp_vae)
    classifier_config = get_config(exp_classifier, key='model_opt')
    vae = init_vae(vae_config['vae_opt'])
    classifier = init_classifier(classifier_config['model_opt'])
    return vae, classifier


def generate_data_4_vae(n_samples, causal, constant_factor, pos_bins, classifier,
                        random_seed=1234):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    imgs = dataset_zip['imgs']
    assert((pos_bins[5][-1] == 31 and constant_factor == [1, 0]) or 
           (pos_bins[5][1] == 31 and constant_factor == [0, 1]))

    final_dict = {}
    for i in range(len(pos_bins)):
        d_sprite_idx, X_true_data, labels = caus_utils.calc_dsprite_idxs(
            num_samples=n_samples, seed=random_seed, constant_factor=constant_factor, 
            causal=causal, color=0, shape=2, scale=5, posXclass_min=pos_bins[i][0], 
            posXclass_max=pos_bins[i][1], posYclass_min=pos_bins[i][2], 
            posYclass_max=pos_bins[i][3])
        D_data = caus_utils.make_dataset_d_sprite(
            d_sprite_dataset=imgs, dsprite_idx=d_sprite_idx, img_size=256)
        D_data_tensor_list = list(map(lambda t: torch.tensor(t).float().unsqueeze(0), D_data))
        D_data_tensor = torch.cat(D_data_tensor_list).unsqueeze(1)
        classes = classifier(D_data_tensor)
        out = torch.nn.Softmax(dim=1)(classes)
        out_argmax = torch.max(out, dim=1)
        final_dict[str(i)] = out_argmax[1]
    return final_dict
    

exp_vae = 'VAE_CausalDsprite_ber_shape2_scale5_ld2'
exp_classifier = 'CausalClassifier'
vae, classifier = load_models(exp_vae, exp_classifier)
posX_bins = [(i, i+3, 0, 31) for i in range(0,31,4)]
posY_bins = [(0, 31, i, i+3) for i in range(0,31,4)]
posX_gt_dict = generate_data_4_vae(100, True, [1,0], posX_bins, classifier)
posY_gt_dict = generate_data_4_vae(100, True, [0,1], posY_bins, classifier)

fixed_codes_dict = sample_latent_codes(2, 100, vae, classifier)

plt.figure(1, figsize=(30, 100))
plt.clf()
plt.suptitle(exp_vae)
# Latent intevention
for i in range(len(fixed_codes_dict.keys())):
    plt.subplot(len(fixed_codes_dict.keys()), len(posX_gt_dict.keys()) + 1, 
                i*(len(posX_gt_dict.keys())+1) + 1)
    plt.hist(fixed_codes_dict[str(i)], bins=64)
    plt.title('vae fixed z' + str(i))
    plt.xlabel('class')

for i in range(len(posX_gt_dict.keys())):
    plt.subplot(len(fixed_codes_dict.keys()), len(posX_gt_dict.keys()) + 1, i + 2)
    plt.hist(posX_gt_dict[str(i)], bins=64)
    plt.title('gt fixed z' + str(i))
    plt.xlabel('class')

for i in range(len(posX_gt_dict.keys())):
    plt.subplot(len(fixed_codes_dict.keys()), len(posX_gt_dict.keys()) + 1, i + len(posX_gt_dict.keys()) + 3)
    plt.hist(posY_gt_dict[str(i)], bins=64)
    plt.title('gt fixed z' + str(i))
    plt.xlabel('class')
    
plt.subplots_adjust(hspace=0.5)
plt.show()



for latent_dim in range(len(fixed_codes_dict)):
    plt.figure(latent_dim, figsize=(30, 100))
    plt.clf()
    plt.suptitle(exp_vae + ' with fixed latent {0} and fixed gt X'.format(str(latent_dim)))
    # Latent intevention
    n_distr = len(posX_gt_dict.keys())
    for i in range(n_distr):
        plt.subplot(2, n_distr/2, 1 + i)
        plt.hist(fixed_codes_dict[str(latent_dim)], bins=64, label='latent', alpha=0.5)
        plt.hist(posX_gt_dict[str(i)], bins=64, label='gt', alpha=0.5)
        plt.legend()
        plt.xlim(0, 64)
        plt.title('fixed gt class' + str(i))
        plt.xlabel('class')
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()


