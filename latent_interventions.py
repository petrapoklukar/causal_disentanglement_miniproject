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

def compute_mmd(sample1, sample2, alpha):
    """
    Computes MMD for samples of the same size bs x n_features using Gaussian
    kernel.
    
    See Equation (3) in http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    for the exact formula.
    """
    # Get number of samples (assuming m == n)
    m = sample1.shape[0]
    
    # Calculate pairwise products for each sample (each row). This yields
    # 2-norms |xi|^2 on the diagonal and <xi, xj> on non diagonal
    xx = torch.mm(sample1, sample1.t())
    yy = torch.mm(sample2, sample2.t())
    zz = torch.mm(sample1, sample2.t())
    
    # Expand the norms of samples into the original size
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    # Remove the diagonal elements, the non diagonal are exactly |xi - xj|^2
    # = <xi, xi> + <xj, xj> - 2<xi, xj> = |xi|^2 + |xj|^2 - 2<xi, xj>
    kernel_samples1 = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    kernel_samples2 = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    kernel_samples12 = torch.exp(- alpha * (rx.t() + ry - 2*zz))
    
    # Normalisations
    n_same = (1./(m * (m-1)))
    n_mixed = (2./(m * m)) 
    
    term1 = n_same * torch.sum(kernel_samples1)
    term2 = n_same * torch.sum(kernel_samples2)
    term3 = n_mixed * torch.sum(kernel_samples12)
    return term1 + term2 - term3

def compute_mmds(latent_dict, posX_gt_dict, posy_gt_dict, alpha):
    scores_dict = {}
    for latent_dim in latent_dict.keys():
        samples_latent = latent_dict[latent_dim].unsqueeze(1).float()
        # scores_dict[latent_dim] = {}
        scores = {}
        for Xgt_dim in posX_gt_dict.keys():
            samples_gt = posX_gt_dict[Xgt_dim].unsqueeze(1).float()
            mmd_score = compute_mmd(samples_latent, samples_gt, alpha)
            scores['X' + Xgt_dim] = round(mmd_score.item(), 2)
    
        for Ygt_dim in posY_gt_dict.keys():
            samples_gt = posY_gt_dict[Ygt_dim].unsqueeze(1).float()
            mmd_score = compute_mmd(samples_latent, samples_gt, alpha)
            scores['Y' + Ygt_dim] = round(mmd_score.item(), 2)
        scores_dict[latent_dim] = scores
    return scores_dict

exp_vae = 'VAE_CausalDsprite_ber_shape2_scale5_ld2'
exp_classifier = 'CausalClassifier'
vae, classifier = load_models(exp_vae, exp_classifier)
posX_bins = [(i, i+3, 0, 31) for i in range(0,31,4)]
posY_bins = [(0, 31, i, i+3) for i in range(0,31,4)]
posX_gt_dict = generate_data_4_vae(100, True, [1,0], posX_bins, classifier)
posY_gt_dict = generate_data_4_vae(100, True, [0,1], posY_bins, classifier)

fixed_codes_dict = sample_latent_codes(2, 100, vae, classifier)

compute_mmds(fixed_codes_dict, posX_gt_dict, posY_gt_dict, alpha=10)





def plot_distributions(exp_vae, fixed_codes_dict, posX_gt_dict, posY_gt_dict):

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
        
        
    for latent_dim in range(len(fixed_codes_dict)):
        plt.figure(5 + latent_dim, figsize=(30, 100))
        plt.clf()
        plt.suptitle(exp_vae + ' with fixed latent {0} and fixed gt Y'.format(str(latent_dim)))
        # Latent intevention
        n_distr = len(posX_gt_dict.keys())
        for i in range(n_distr):
            plt.subplot(2, n_distr/2, 1 + i)
            plt.hist(fixed_codes_dict[str(latent_dim)], bins=64, label='latent', alpha=0.5)
            plt.hist(posY_gt_dict[str(i)], bins=64, label='gt', alpha=0.5)
            plt.legend()
            plt.xlim(0, 64)
            plt.title('fixed gt class' + str(i))
            plt.xlabel('class')
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
    
