import os
import time
import torch
import numpy as np
from hparam import hparam as hp
from speech_embedder_net import get_cossim, R2Plus1DNet

def extract(model_path,mode):
    device = torch.device("cpu")
    embedder_net = R2Plus1DNet([2,2,2,2]).to(device)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    if mode == "centroid":
        path = hp.data.train_path
        save_dir = hp.data.centroid_dir
    else:
        path = hp.data.test_path
        save_dir = hp.data.embedding_dir
    np_file_dir = [b for b in os.listdir(path) if b[0] != "."]
    os.makedirs(save_dir, exist_ok=True)
    for d in np_file_dir:
        np_file_list = os.listdir(os.path.join(path,d))
        buffer = []
        for f in np_file_list:
            utter = np.load(os.path.join(path,d,f))
            utter = np.resize(utter,(1,)+utter.shape)
            utter = torch.tensor(utter).to(device)

            buffer.append(embedder_net(utter)[0].detach())

        buffer = np.stack(buffer)
        if mode == "centroid":
            buffer = buffer.mean(axis=0)
        np.save(os.path.join(save_dir,d)+"/{}_{}.npy".format(d,mode),buffer)

if __name__ == "__main__":
    extract(hp.model.model_path,hp.data.extract)