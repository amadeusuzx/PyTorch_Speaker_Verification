#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec

class SpeakerDatasetTIMIT(Dataset):
    
    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        shuffle(self.speakers)
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)

class SpeakerDatasetTIMITPreprocessed(Dataset):
    
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
        np_file_list = os.listdir(os.path.join(self.path+selected_dir))
        selected_file = random.sample(np_file_list,self.utter_num)
        utters = []
        for s in selected_file:
            frames = np.load(os.path.join(selected_dir,s))
            utters.append(self.crop(frames,32))
        utters = torch.tensor(np.array(utters))
        return utters
    def crop(self, buffer, clip_len):
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        buffer = buffer[:,time_index:time_index + clip_len,:,:]
        return buffer   