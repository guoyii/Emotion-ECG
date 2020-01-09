import torch
from MLP import MLP
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

weights_path = "../output/final_result/Network_final.pth.gz"
test_set = MySet(parser.txt_path, mode="test")
net = MLP(parser)

weights = torch.load(weights_path,map_location='cuda:%d'%(gpu_id)) #GPU ###
net.load_state_dict(weights)
net.cuda().eval() #测试模式 GPU


test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

start=time.time()
labels, predicts = [], []
with torch.no_grad():
    for batch_idx, (feature, label) in enumerate(test_loader):
        feature = feature.float().cuda()  #GPU
        label = label.data.numpy()
        predict = net(feature)
        predict = predict.data.cpu().numpy()
        predict = np.argmax(predict,axis=1)
        labels.extend(list(label))
        predicts.extend(list(predict))
    acc = accuracy_score(labels, predicts)
end=time.time()
print("test time:{:.2f}\n".format(end-start))
print("Total acc :{:.4f}\n".format(acc*100))



