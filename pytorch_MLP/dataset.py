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
        feature_path = data_item["feature"]
        label = data_item["label"]
        feature = loadmat(feature_path)  #读取mat文件
        feature = feature['temp']

        #  由于不需要数据增强，故训练测试一致
        if self.mode =="train":
            pass
        
        label = np.array(label, dtype=np.int64)
        feature_tensor = torch.from_numpy(feature)
        label_tensor = torch.from_numpy(label)

        return feature_tensor, label_tensor

    def __len__(self):
        return len(self.data_list)


def read_list(mode, txt_path):
    if mode == "train":
        txt_read_path = os.path.join(txt_path, "train.txt")
    elif mode == "val":
        txt_read_path = os.path.join(txt_path, "val.txt")
    elif mode == "test":
        txt_read_path = os.path.join(txt_path, "test.txt")
    else:
        raise ValueError

    fid = open(txt_read_path, "r")
    lines = fid.readlines()
    fid.close()
    random.shuffle(lines)  #打乱顺序

    data_list = []
    for line in lines:
        feature_name, label = line.split(" ")#指定空格键为分隔符
        feature_info = {
            "feature": feature_name,
            "label": int(label)
        }
        data_list.append(feature_info)

    return data_list