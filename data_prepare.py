from glob import glob
import numpy as np
import cv2
import os
from hparam import hparam as hp
def data_prep(dataset,tr_path,ts_path):
    os.makedirs(tr_path, exist_ok=True)
    os.makedirs(ts_path, exist_ok=True)
    dataset = dict()
    
    for c in commands:
        dataset[c] = []
        for f in sorted(glob("./data_kimura_10words/"+c+"*")):
            capture = cv2.VideoCapture(f)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            sample = np.linspace(0, frame_count-1, num=16, dtype=int)
            if frame_count>32:

                buffer = np.empty((16,80,100, 3), np.dtype('float32'))
                count = 0
                retaining = True
                j=0
                while (count < frame_count and retaining and j<16):
                    retaining, frame = capture.read()
                    if count == sample[j]:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        buffer[j] = frame
                        j+=1
                    count += 1
                
                buffer = buffer.transpose((3, 0, 1, 2)) 
                buffer = (buffer - 128)/128
                dataset[c].append(buffer)
    for c in commands:
        np.save(os.path.join(tr_path,c)+".npy",np.stack(dataset[c])[:-10])
        np.save(os.path.join(ts_path,c)+".npy",np.stack(dataset[c])[-10:])
if __name__ == "__main__":
    commands = ["black","cancel","centeralign","copy","large","medium","newslide","paste","red","textbox"]
    data_prep(hp.data.dataset,hp.data.train_path,hp.data.test_path)

