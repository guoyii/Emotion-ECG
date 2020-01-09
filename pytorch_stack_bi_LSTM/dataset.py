from torch.utils.data import Dataset
from scipy.io import loadmat
from numpy import random
import os
import numpy as np
import torch

class MySet(Dataset):
    def __init__(self, txt_path, mode="train"):
        self.mode = mode
        self.data_list = read_list(self.mode, txt_path)

    def __getitem__(self, item):
        data_item = self.data_list[item]
        sequence_path = data_item["sequence"]
        name = sequence_path
        label = data_item["label"]
        sequence = loadmat(sequence_path)  #读取mat文件
        sequence = sequence['temp']
        if sequence.shape[0] < 800:
            sequence = np.vstack((sequence,np.zeros((800-sequence.shape[0],1))))
        else:
            sequence = sequence[0:800]



        #  由于不需要数据增强，故训练测试一致
        if self.mode =="train":
            pass
        
        label = np.array(label, dtype=np.int64)
        sequence_tensor = torch.from_numpy(sequence)
        label_tensor = torch.from_numpy(label)

        return sequence_tensor, label_tensor, name

    def __len__(self):
        return len(self.data_list)


def read_list(mode, txt_path):
    if mode == "train":
        txt_read_path = os.path.join(txt_path, "train.txt")
    elif mode == "val":
        txt_read_path = os.path.join(txt_path, "val.txt")
    elif mode == "test":
        txt_read_path = os.path.join(txt_path, "test.txt")
    elif mode == "all":
        txt_read_path = os.path.join(txt_path, "all_data.txt")
    else:
        raise ValueError

    fid = open(txt_read_path, "r")
    lines = fid.readlines()
    fid.close()
    random.shuffle(lines)  #打乱顺序

    data_list = []
    for line in lines:
        sequence_name, label = line.split(" ")#指定空格键为分隔符
        sequence_info = {
            "sequence": sequence_name,
            "label": int(label)
        }
        data_list.append(sequence_info)

    return data_list