import os
from scipy.io import loadmat       
import numpy.fft as fft     
import numpy as np
import pywt



time = 30                            
sample_rate = 360                   
data_length = 800


def read_dataPath(data_path):
    X = [] 
    labels = [] 
    for i, data_folder in enumerate(os.listdir(data_path)):
        for data in os.listdir(data_path +"/" + data_folder):
            if data[len(data)-4:len(data)] == ".mat":                   
                X_path_one = data_path + "/" + data_folder + "/" + data
                labels_one = data[data.find("_")+1]                       
                X.append(X_path_one)
                labels.append(labels_one)

    temp = np.array([X, labels])
    temp = temp.transpose()                                              
    np.random.shuffle(temp)                                                 
    
    X = list(temp[:, 0])                                              
    labels = list(temp[:, 1])
    
    return X, labels


def read_data(X, labels):
    # X_src = np.random.rand(len(X), time*sample_rate).astype("float32")   #
    X_src = np.zeros((len(X), data_length), np.float64)
    # label_src = np.random.rand(len(labels), 0)
    for i in range(len(X)):
        data_temp = (loadmat(X[i])["temp"]).flatten()                        
        if data_temp.shape[0]>data_length:
            data_temp = data_temp[:data_length]
        X_src[i][0: data_temp.shape[0]] = data_temp
        label_temp = int(labels[i])                             
        if label_temp == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    return X_src, labels



##********************************************************************************************************
def myfft(X):
    X_fft = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        complex_y = fft.fft(X[i])                                        
        abs_y = abs(complex_y)                                          
        rael_y = complex_y.real                                        
        imag_y = complex_y.imag                                        

        X_fft[i] = abs_y                                            
    return X_fft



##********************************************************************************************************
def mydwt(X):
    X_dwt = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        (cA, cD) = pywt.dwt(X[i], 'db1')
        X_dwt[i][:int(data_length/2)] = cA.flatten()
        X_dwt[i][int(data_length/2):data_length] = cD.flatten()
    return X_dwt



##********************************************************************************************************
def mywavedec(X, level = 2):
    X_wavedec = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        cA2, cD2, cD1 = pywt.wavedec(X[i], 'db1', level=level)
        X_wavedec[i][:int(data_length/2)] = cD1.flatten()
        X_wavedec[i][int(data_length/2):int(int(data_length/2)+int(data_length/2)/2)] = cD2.flatten()
        X_wavedec[i][int(int(data_length/2)+int(data_length/2)/2):data_length] = cA2.flatten()
    return X_wavedec



##********************************************************************************************************
def read_self_dataPath(data_path):
    X = [] 
    labels = [] 
    for i, data_folder in enumerate(os.listdir(data_path)):
        for data in os.listdir(data_path +"/" + data_folder):
            if data[len(data)-4:len(data)] == ".mat":                 
                X_path_one = data_path + "/" + data_folder + "/" + data
                if "fear" in data_folder:
                    labels_one = 0
                elif "joy" in data_folder:
                    labels_one = 1
                X.append(X_path_one)
                labels.append(labels_one)
    temp = np.array([X, labels])
    temp = temp.transpose()                                            
    np.random.shuffle(temp)                                              
    
    X = list(temp[:, 0])                                             
    labels = list(temp[:, 1])
    # labels = [int(i) for i in labels]
    
    return X, labels


##********************************************************************************************************
def read_self_data(X, labels):
    # X_src = np.random.rand(len(X), time*sample_rate).astype("float32")   
    X_src = np.zeros((len(X), data_length), np.float64)
    # label_src = np.random.rand(len(labels), 0)
    for i in range(len(X)):
        data_temp = (loadmat(X[i])["temp"]).flatten()                        
        if data_temp.shape[0]>data_length:
            data_temp_new = data_temp[:data_length]
        else:
            data_temp_new = data_temp

        X_src[i][0: data_temp_new.shape[0]] = data_temp_new
        label_temp = int(labels[i])                                      
        if label_temp == 1:
            labels[i] = 1
        else:
            labels[i] = 0
    return X_src, labels