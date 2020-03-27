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
import sys
import pickle


def preprocess_causal_data(filename):
    with open('datasets/'+filename, 'rb') as f:
        if sys.version_info[0] < 3:
            data_list = pickle.load(f)
        else:
            data_list = pickle.load(f, encoding='latin1')
        #data_list = pickle.load(f)

    random.seed(2610)

    random.shuffle(data_list)

    splitratio = int(len(data_list) * 0.15)
    train_data = data_list[splitratio:]
    test_data = data_list[:splitratio]

    train_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    train_data))
    test_data1 = list(map(lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1),
                                torch.tensor(t[1]/255).float().permute(2, 0, 1),
                                torch.tensor(t[2]).float()),
                    test_data))
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
       
        if split == 'test':
            with open('datasets/test_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open('datasets/train_'+self.dataset_name+'.pkl', 'rb') as f:
                self.data = pickle.load(f)


    def __getitem__(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)



    