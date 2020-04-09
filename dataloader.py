#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:15:30 2019

@author: petrapoklukar

"""
from __future__ import print_function
import torch
import torch.utils.data as data
import random
import pickle
import numpy as np

def preprocess_causal_data(filename):
    with open('datasets/' + filename, 'rb') as f:
        data_list = pickle.load(f)

    random.seed(2610)
    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]
    if filename == 'causal_dsprite_shape2_scale5_imgs.pkl':
        train_data1 = list(map(
            lambda t: torch.tensor(t).float().unsqueeze(0), train_data))
        test_data1 = list(map(
            lambda t: torch.tensor(t).float().unsqueeze(0), test_data))
    else:
        train_data1 = list(map(
            lambda t: torch.tensor(t/255.).float().permute(2, 0, 1), train_data))
        test_data1 = list(map(
            lambda t: torch.tensor(t/255.).float().permute(2, 0, 1), test_data))
        
    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class CausalDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
       
        name = 'datasets/{0}_{1}.pkl'.format(self.split, self.dataset_name)
        with open(name, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        img = self.data[index]
        return img
    
    def get_subset(self, max_ind, n_points):
        self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        subset_list = [self.data[i].numpy() for i in self.prd_indices]
        return np.array(subset_list).reshape(n_points, -1)

    def __len__(self):
        return len(self.data)



    