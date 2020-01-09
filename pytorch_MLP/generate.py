import os
import re
import numpy as np
from numpy import random
from utils import check_dir
from scipy.io import loadmat
from scipy.io import savemat

#联合频域和LSTM特征再保存
LSTM_feature_path = "../LSTM_feature"
frequency_feature_path = "../frequency_feature/out"
stack_feature_path = "../stack_feature"
check_dir(stack_feature_path)
sample_list = os.listdir(LSTM_feature_path)

for sample in sample_list:
    loaded_sample_LSTM = loadmat(os.path.join(LSTM_feature_path,sample))
    LSTM_feature = loaded_sample_LSTM['temp']
    loaded_sample_frequency = loadmat(os.path.join(frequency_feature_path,sample[0:sample.index('.')]+'_frequency.mat'))
    frequency_feature = loaded_sample_frequency['data']
    stack_feature = np.hstack((LSTM_feature,frequency_feature))
    savemat(os.path.join(stack_feature_path,sample),{'temp':stack_feature})



path = '../stack_feature'
output_path = '../txt/stack_feature'
check_dir(path)
check_dir(output_path)

data = []

for root, dirs, files in os.walk(path):
    for name in files:
        if name[0:3] == 'joy':  #正常和非正常
            data.append([os.path.join(root, name),'1'])
        else:
            data.append([os.path.join(root, name),'0'])

random.shuffle(data) #打乱顺序

train_file=open(os.path.join(output_path, "train.txt"), "w")
val_file=open(os.path.join(output_path, "val.txt"), "w")
test_file=open(os.path.join(output_path, "test.txt"), "w")

l = len(data)

#train
for sequence in data[0:int(0.6*l)]:
    train_file.write("{} {}\n".format(sequence[0],sequence[1]))
#val
for sequence in data[int(0.6*l):int(0.8*l)]:
    val_file.write("{} {}\n".format(sequence[0],sequence[1]))
#test
for sequence in data[int(0.8*l):]:
    test_file.write("{} {}\n".format(sequence[0],sequence[1]))