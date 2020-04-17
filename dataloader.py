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
    if 'dsprite' in filename:
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
        
        
def preprocess_causal_classifier_data(filename):
    with open('datasets/' + filename, 'rb') as f:
        data = pickle.load(f)
        data_list = data['data']
        label_list = list(data['labels'].astype(int))
        
    zipped_list = list(zip(data_list, label_list))
    random.seed(2610)
    random.shuffle(zipped_list)

    splitratio = int(len(zipped_list) * 0.15)
    train_data = zipped_list[splitratio:]
    test_data = zipped_list[:splitratio]
    if 'dsprite' in filename:
        train_data1 = list(map(
            lambda t: (torch.tensor(t[0]).float().unsqueeze(0), 
                       torch.tensor(t[1])), train_data))
        test_data1 = list(map(
            lambda t: (torch.tensor(t[0]).float().unsqueeze(0), 
                       torch.tensor(t[1])), test_data))
    else:
        train_data1 = list(map(
            lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1), 
                       torch.tensor(t[1])), train_data))
        test_data1 = list(map(
            lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1), 
                       torch.tensor(t[1])), test_data))
    print('Train and test split lengths:', len(train_data1), len(test_data1))
    print('An element is of type ', type(train_data1[0]))

    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


# ----------------------- #
# --- Custom Datasets --- #
# ----------------------- #
class CausalDataset(data.Dataset):
    def __init__(self, dataset_name, split, subset_classes=[]):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
       
        name = 'datasets/{0}_{1}.pkl'.format(self.split, self.dataset_name)
        with open(name, 'rb') as f:
            self.data = pickle.load(f)
        
        if subset_classes:
            self.get_subset_classes(subset_classes)

    def __getitem__(self, index):
        img = self.data[index]
        return img
    
    def get_subset_classes(self, classes):
        subset = []
        for i in range(len(self.data)):
            if self.data[i][1] in classes:
                subset.append(self.data[i]) 
        self.data = subset
    
    def get_subset(self, max_ind, n_points):
        self.prd_indices = np.random.choice(max_ind, n_points, replace=False)
        subset_list = [self.data[i].numpy() for i in self.prd_indices]
        return np.array(subset_list).reshape(n_points, -1)

    def __len__(self):
        return len(self.data)



    