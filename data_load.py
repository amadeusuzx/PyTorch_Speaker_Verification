#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
from hparam import hparam as hp


class TrainDataset(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
          self.path = hp.data.train_path
          self.utter_num = hp.train.M
        else:
          self.path = hp.data.test_path
          self.utter_num = hp.test.M

        self.file_list = [b for b in os.listdir(self.path) if b[0] != "."]
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):


        np_file_dir = [b for b in os.listdir(self.path) if b[0] != "."]
        selected_file = random.sample(np_file_dir, 1)[0]  # select random speaker          
        utterances = np.load(selected_file)

        index = np.random.randint(0, utterances.shape[0], self.utter_num)   # select M utterances per speaker
        selected_utters = torch.tensor(utterances[index])

        return selected_utters
    def crop(self, buffer, clip_len):
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        buffer = buffer[:,time_index:time_index + clip_len,:,:]
        return buffer   


class TestDataset(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
 
        self.path = hp.data.embedding_path
        self.utter_num = hp.test.M
        self.file_list = [b for b in os.listdir(self.path) if b[0] != "."]
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_dir = os.listdir(self.path)
        selected_file = random.sample(np_file_dir, 1)[0]  # select random speaker          
        embs = np.load(selected_file)
        
        index = np.random.randint(0, embs.shape[0], self.utter_num)   # select M utterances per speaker
        selected_embs = torch.tensor(embs[index])
        return selected_embs 