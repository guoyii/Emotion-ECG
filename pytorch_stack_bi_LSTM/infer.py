import torch
from stack_bi_lstm import Stack_Bi_LSTM
from init import InitParser
from dataset import MySet
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import accuracy_score


gpu_id = 0
batch_size = 1000

torch.cuda.set_device(gpu_id)

parser = InitParser()
net = Stack_Bi_LSTM(parser)
weights_path = "../output/Network_199_0.9587.pth.gz"
weights = torch.load(weights_path,map_location='cuda:%d'%(gpu_id)) #GPU ###
net.load_state_dict(weights)
net.cuda().eval() #测试模式 GPU


test_set = MySet(parser.txt_path, parser.data_path, mode="test")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

start=time.time()
labels, predicts = [], []
with torch.no_grad():
    for batch_idx, (sequence, label) in enumerate(test_loader):
        sequence = sequence.float().cuda()  #GPU
        label = label.data.numpy()
        predict = net(sequence.permute(1,0,2))
        predict = predict.data.cpu().numpy()
        predict = np.argmax(predict,axis=1)
        labels.extend(list(label))
        predicts.extend(list(predict))
    acc = accuracy_score(labels, predicts)
end=time.time()
print("test time:{:.2f}\n".format(end-start))
print("Total acc :{:.4f}\n".format(acc*100))



