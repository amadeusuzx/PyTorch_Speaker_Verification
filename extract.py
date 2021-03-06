import os
import time
import torch
import numpy as np
from hparam import hparam as hp
from speech_embedder_net import get_cossim, R2Plus1DNet
import sys
def extract(model_path,dataset):
    device = torch.device("cpu")
    embedder_net = R2Plus1DNet([2,2,2,2]).to(device)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()
    save_path = os.path.join(hp.data.embedding_path + dataset)
    if dataset == "train":
        path = hp.data.train_path
    else:
        path = hp.data.test_path
    np_file_dir = [b for b in os.listdir(path) if b[0] != "."]
    os.makedirs(save_path, exist_ok=True)
    for d in np_file_dir:
        np_file_list = os.listdir(os.path.join(path,d))
        buffer = []
        for f in np_file_list:
            utter = np.load(os.path.join(path,d,f))
            utter = np.resize(utter,(1,)+utter.shape)
            utter = torch.tensor(utter).to(device)

            buffer.append(embedder_net(utter)[0].detach())

        buffer = np.stack(buffer)
        np.save(os.path.join(save_path,d) + ".npy",buffer)

if __name__ == "__main__":
    dataset = sys[1]
    extract(hp.model.model_path,dataset)