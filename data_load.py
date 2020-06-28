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
        selected_dir = random.sample(np_file_dir, 1)[0]  # select random speaker          
        np_file_list = os.listdir(os.path.join(self.path,selected_dir))
        selected_file = random.sample(np_file_list,self.utter_num)
        utters = []
        for s in selected_file:
            frames = np.load(os.path.join(self.path,selected_dir,s))
            utters.append(frames)
        utters = torch.tensor(np.array(utters))
        return utters
    def crop(self, buffer, clip_len):
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        buffer = buffer[:,time_index:time_index + clip_len,:,:]
        return buffer   


class TestDataset(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
 
        self.path = hp.data.test_path
        self.utter_num = hp.test.M
        self.file_list = [b for b in os.listdir(self.path) if b[0] != "."]
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_dir = [b for b in os.listdir(self.path) if b[0] != "."]
        selected_dir = random.sample(np_file_dir, 1)[0]  # select random speaker          
        np_file_list = [b for b in os.listdir(os.path.join(self.path,selected_dir)) if "emb" in b]
        selected_file = random.sample(np_file_list,self.utter_num)
        embs = []
        for s in selected_file:
            emb = np.load(os.path.join(self.path,selected_dir,s))
            embs.append(emb[0])
        embs = torch.tensor(np.array(embs))
        return embs
