%%writefile ./PyTorch_Speaker_Verification/extract.py
import os
import random
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import  GE2ELoss, get_centroids, get_cossim, R2Plus1DNet

def extract(model_path):
    device = torch.device(hp.device)
    crop = SpeakerDatasetTIMITPreprocessed.crop
    embedder_net = R2Plus1DNet([2,2,2,2]).to(device)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    path = hp.data.test_path

    np_file_dir = [b for b in os.listdir(path) if b[0] != "."]

    for d in np_file_dir:
        i=0
        np_file_list = os.listdir(os.path.join(path,d))
        for f in np_file_list:
            i+=1
            utter = np.load(os.path.join(path,d,f))
            utter = np.resize(utter,(1,)+utter.shape)
            utter = torch.tensor(utter)

            utter = utter.to(device)
            np.save(os.path.join(path,d)+"/emb%d"%i,embedder_net(utter).detach().cpu())

if __name__ == "__main__":
    extract(hp.model.model_path)