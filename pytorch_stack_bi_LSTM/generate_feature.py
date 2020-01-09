import torch
from stack_bi_lstm import Stack_Bi_LSTM
from init import InitParser
from dataset import MySet
from torch.utils.data import DataLoader
import time
import numpy as np
from scipy.io import savemat
import os,re
from utils import check_dir

gpu_id = 0
batch_size = 1000

torch.cuda.set_device(gpu_id)

parser = InitParser()
net = Stack_Bi_LSTM(parser)
weights_path = "../output/fine_tuning_result/Network_fine_tuning.pth.gz"
weights = torch.load(weights_path,map_location='cuda:%d'%(gpu_id)) #GPU ###
net.load_state_dict(weights)
net.cuda().eval() #测试模式 GPU


all_data_set = MySet(parser.fine_tuning_txt_path , mode="all")
all_data_loader = DataLoader(all_data_set, batch_size=batch_size, shuffle=True)

output_path = '../LSTM_feature'
check_dir(output_path)

with torch.no_grad():
    for batch_idx, (sequence, label, name) in enumerate(all_data_loader):
        sequence = sequence.float().cuda()  #GPU
        label = label.data.numpy()
        predict,feature = net(sequence.permute(1,0,2))

feature = feature.data.cpu().numpy()
for i in range(feature.shape[0]):
    savemat(os.path.join(output_path,re.search('[a-z]*_[0-9]*.mat',name[i]).group()),{'temp':feature[i,:]})
