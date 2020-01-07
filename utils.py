import os
from scipy.io import loadmat           ## 用于加载mat文件
import numpy.fft as fft                ## 用于傅里叶变换
import numpy as np
import pywt



time = 30                              ## 数据长度
sample_rate = 360                      ## 采样频率 用于计算单个样本维度 time*sample_rate
data_length = 800

## 读取各数据文件路径，并随机打乱
##********************************************************************************************************
def read_dataPath(data_path):
    X = [] 
    labels = [] 
    for i, data_folder in enumerate(os.listdir(data_path)):
        print("读取第{}个数据路径...............".format(i))
        for data in os.listdir(data_path +"/" + data_folder):
            if data[len(data)-4:len(data)] == ".mat":                       ## 只需要读取30秒数据
                X_path_one = data_path + "/" + data_folder + "/" + data
                labels_one = data[data.find("_")+1]                         ## label根据命名规则确定
                X.append(X_path_one)
                labels.append(labels_one)

    temp = np.array([X, labels])
    temp = temp.transpose()                                                 ## 转置
    np.random.shuffle(temp)                                                 ## 打乱顺序，可删除
    
    X = list(temp[:, 0])                                                    ## 重新获取labels与X(这里问路径名称)
    labels = list(temp[:, 1])
    # labels = [int(i) for i in labels]
    
    return X, labels


## 读取数据并保存
##********************************************************************************************************
def read_data(X, labels):
    # X_src = np.random.rand(len(X), time*sample_rate).astype("float32")   ## time*sample_rate 360：采样率 30：时间  用于存放心电数据
    X_src = np.zeros((len(X), data_length), np.float64)
    # label_src = np.random.rand(len(labels), 0)
    for i in range(len(X)):
        print("加入第{}个个体数据...............".format(i)) 
        data_temp = (loadmat(X[i])["temp"]).flatten()                           ## 读入mat数据 flatten()将列转为行，可根据实际需要删除
        if data_temp.shape[0]>data_length:
            data_temp = data_temp[:data_length]
        X_src[i][0: data_temp.shape[0]] = data_temp
        label_temp = int(labels[i])                                         ## label无变化，可删除，可根据label命名修改
        if label_temp == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    return X_src, labels


## 傅里叶变换
##********************************************************************************************************
def myfft(X):
    X_fft = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        print("对第{}个个体数据进行傅里叶变换...............".format(i))
        complex_y = fft.fft(X[i])                                        ## 复数形式
        abs_y = abs(complex_y)                                           ## 取模
        rael_y = complex_y.real                                          ## 取实部
        imag_y = complex_y.imag                                          ## 虚部

        X_fft[i] = abs_y                                                 ## 根据需要选择
    return X_fft


## 小波变换
##********************************************************************************************************
def mydwt(X):
    X_dwt = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        print("对第{}个个体数据进行小波变换...............".format(i))
        (cA, cD) = pywt.dwt(X[i], 'db1')
        X_dwt[i][:int(data_length/2)] = cA.flatten()
        X_dwt[i][int(data_length/2):data_length] = cD.flatten()
    return X_dwt


## 小波分解
##********************************************************************************************************
def mywavedec(X, level = 2):
    X_wavedec = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        print("对第{}个个体数据进行小波变换...............".format(i))
        cA2, cD2, cD1 = pywt.wavedec(X[i], 'db1', level=level)
        X_wavedec[i][:int(data_length/2)] = cD1.flatten()
        X_wavedec[i][int(data_length/2):int(int(data_length/2)+int(data_length/2)/2)] = cD2.flatten()
        X_wavedec[i][int(int(data_length/2)+int(data_length/2)/2):data_length] = cA2.flatten()
    return X_wavedec



## 读取各数据文件路径，并随机打乱
##********************************************************************************************************
def read_self_dataPath(data_path):
    X = [] 
    labels = [] 
    for i, data_folder in enumerate(os.listdir(data_path)):
        print("读取第{}个数据路径...............".format(i))
        for data in os.listdir(data_path +"/" + data_folder):
            if data[len(data)-4:len(data)] == ".mat":                       ## 只需要读取30秒数据
                X_path_one = data_path + "/" + data_folder + "/" + data
                if "fear" in data_folder:
                    labels_one = 0
                elif "joy" in data_folder:
                    labels_one = 1
                X.append(X_path_one)
                labels.append(labels_one)
    temp = np.array([X, labels])
    temp = temp.transpose()                                                 ## 转置
    np.random.shuffle(temp)                                                 ## 打乱顺序，可删除
    
    X = list(temp[:, 0])                                                    ## 重新获取labels与X(这里问路径名称)
    labels = list(temp[:, 1])
    # labels = [int(i) for i in labels]
    
    return X, labels


## 读取数据并保存
##********************************************************************************************************
def read_self_data(X, labels):
    # X_src = np.random.rand(len(X), time*sample_rate).astype("float32")   ## time*sample_rate 360：采样率 30：时间  用于存放心电数据
    X_src = np.zeros((len(X), data_length), np.float64)
    # label_src = np.random.rand(len(labels), 0)
    for i in range(len(X)):
        print("加入第{}个个体数据...............".format(i)) 
        data_temp = (loadmat(X[i])["temp"]).flatten()                           ## 读入mat数据 flatten()将列转为行，可根据实际需要删除
        if data_temp.shape[0]>data_length:
            data_temp_new = data_temp[:data_length]
        else:
            data_temp_new = data_temp

        X_src[i][0: data_temp_new.shape[0]] = data_temp_new
        label_temp = int(labels[i])                                         ## label无变化，可删除，可根据label命名修改
        if label_temp == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    return X_src, labels