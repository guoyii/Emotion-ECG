from init import InitParser
from utils import AvgMeter,plot,check_dir
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from stack_bi_lstm import Stack_Bi_LSTM
from dataset import MySet
import numpy as np
import os,time

# 验证epoch, 返回验证loss和acc
def val_epoch(net,loader,cost):
    net.eval()
    labels, predicts = [],[]
    loss_meter = AvgMeter()
    for batch_idx, (sequence, label) in enumerate(loader):
        sequence = sequence.float().cuda() #GPU
        label = label.float().cuda()  #GPU
        predict = net(sequence.permute(1,0,2))
        loss = cost(predict,label.long())
        loss_meter.update(loss.item())
        predict = predict.data.cpu().numpy()
        label = label.data.cpu().numpy()
        predicts.extend(list(np.argmax(predict,axis = 1)))
        labels.extend(list(label))
    acc = accuracy_score(labels,predicts)
    return loss_meter.avg, acc

# 训练epoch，返回训练loss和acc
def train_epoch(net,loader,optimizer,cost):
    net.train()
    labels, predicts = [],[]
    batch_acc=[]
    loss_meter = AvgMeter()
    t_1 = time.time()
    for batch_idx, (sequence, label) in enumerate(loader):
        sequence = sequence.float().cuda()  #GPU
        label = label.float().cuda()  #GPU
        optimizer.zero_grad()
        predict = net(sequence.permute(1,0,2))
        loss = cost(predict, label.long())
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        predict = predict.data.cpu().numpy()
        label = label.data.cpu().numpy()
        predicts.extend(list(np.argmax(predict,axis = 1)))
        labels.extend(list(label))
        acc = accuracy_score(labels,predicts)
        batch_acc.append(acc)
        if batch_idx % 10 == 0:
            info = [batch_idx, loss_meter.val,acc,time.time()-t_1]
            t_1 = time.time()
            print("Batch: {} Loss: {:.4f} Batch_acc: {:0.4f} Time:{:0.2f}\n".format(*info), end="")
    t_acc=np.array(batch_acc)
    return loss_meter.avg, acc

def main(args):
    #建立文件夹
    if args.is_fine_tuning:
        result_path = os.path.join(args.output_path,"fine_tuning_result")
    else:
        result_path = os.path.join(args.output_path, "result")
    check_dir(args.output_path)
    check_dir(result_path)

    #设置gpu
    torch.cuda.set_device(args.gpu_id) #GPU

    #加载数据预训练数据或微调数据
    if args.is_fine_tuning:
        train_set = MySet(args.fine_tuning_txt_path, args.data_path, mode="train")
        val_set = MySet(args.fine_tuning_txt_path, args.data_path, mode="val")
        train_loader = DataLoader(train_set, batch_size=args.fine_tuning_batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.fine_tuning_batch_size, shuffle=False)
    else:
        train_set = MySet(args.txt_path, args.data_path, mode="train")
        val_set = MySet(args.txt_path, args.data_path, mode="val")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    #网络模型
    net = Stack_Bi_LSTM(args)
    net.cuda()  #GPU

    #使用Adam优化器和交叉熵损失函数 判断是否微调
    if args.is_fine_tuning:
        weights_path = "../ouput/Network_199_0.9587.pth.gz"
        weights = torch.load(weights_path,map_location = "cuda:%d"%(args.gpu_id))
        net.load_state_dict(weights)
        for param in list(net.parameters())[0:24]: #前面3(layers)*2(bi-direction)*4(LSTM)
            param.require_grad = False
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters())) #使用默认配置
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    cost = torch.nn.CrossEntropyLoss()
    epoch_loss,epoch_acc,val_loss,val_acc =[],[],[],[]

    #epoch循环
    t0=time.time()
    for epoch in range(args.num_epoch):
        t_loss,t_acc = train_epoch(net,train_loader,optimizer,cost)
        epoch_loss.append(t_loss)
        epoch_acc.append(t_acc)

        v_loss,v_acc= val_epoch(net, val_loader,cost) 
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        #画出训练集和测试集的loss及acc
        if args.is_fine_tuning:
            plot(epoch_acc,val_acc,result_path,'train_acc','val_acc',args.fine_tuning_batch_size,'acc',pre_train=False)
            plot(epoch_loss,val_loss,result_path,'train_loss','val_loss',args.fine_tuning_batch_size,'loss',pre_train = False)
        else:
            plot(epoch_acc,val_acc,result_path,'train_acc','val_acc',args.batch_size,'acc')
            plot(epoch_loss,val_loss,result_path,'train_loss','val_loss',args.batch_size,'loss')

            info = [str(epoch).zfill(3), t_loss, v_acc]
            print("Epoch: {} | train Loss: {:.4f} val ACC: {:.4f}".format(*info))

    t1=time.time()
    print("Optimization Finished!  Cost time:{:.1f} minutes".format((t1-t0)/60))
    print("The final acc=%g"%v_acc)

    #保存最终模型
    state=net.state_dict()
    torch.save(state, os.path.join(args.output_path, "Network_{}_{:.4f}.pth.gz".format(epoch,v_acc)))

if __name__ == '__main__':
    parsers = InitParser()
    main(parsers)