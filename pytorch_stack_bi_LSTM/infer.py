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
parser = InitParser()

torch.cuda.set_device(gpu_id)

if parser.is_fine_tuning:
    weights_path = "../output/fine_tuning_result/Network_fine_tuning.pth.gz"
    test_set = MySet(parser.fine_tuning_txt_path, mode="test")
else:
    weights_path = "../output/pretain_result/Network_pretain.pth.gz"
    test_set = MySet(parser.pretain_txt_path, mode="test")

net = Stack_Bi_LSTM(parser)

weights = torch.load(weights_path,map_location='cuda:%d'%(gpu_id)) #GPU ###
net.load_state_dict(weights)
net.cuda().eval() #测试模式 GPU


test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

start=time.time()
labels, predicts = [], []
with torch.no_grad():
    for batch_idx, (sequence, label, _) in enumerate(test_loader):
        sequence = sequence.float().cuda()  #GPU
        label = label.data.numpy()
        predict,_ = net(sequence.permute(1,0,2))
        predict = predict.data.cpu().numpy()
        predict = np.argmax(predict,axis=1)
        labels.extend(list(label))
        predicts.extend(list(predict))
    acc = accuracy_score(labels, predicts)
end=time.time()
print("test time:{:.2f}\n".format(end-start))
print("Total acc :{:.4f}\n".format(acc*100))



