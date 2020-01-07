from sklearn.decomposition import PCA  ## 用于PCA降维，需要一起训练
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
                                       ## 标准化处理
from utils import *
import numpy as np
import numpy.fft as fft 
import pywt
import h5py
import os
from scipy.io import loadmat
import scipy.io as sio
import matplotlib.pylab as plt 
from sklearn.externals import joblib

n_components = 50
data_length = 800
model_pca_name = "model_pca_fft+dwt_"

result_path = "V:/users/gy/MyProject/ECG/result"
data_path = "V:/users/gy/data/ECG/Self"
model_pca_path = result_path + "/model_pca"
out_path = result_path + "/out"

if os.path.exists(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl"):
    model_pca = joblib.load(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl")
else:
    print("No PCA model.....")

def preprocess(X_src):
    X_fft = abs(fft.fft(X_src))

    X_dwt = np.zeros(data_length, np.float64)
    (cA, cD) = pywt.dwt(X_src, 'db1')
    X_dwt[i][:int(data_length/2)] = cA.flatten()
    X_dwt[i][int(data_length/2):data_length] = cD.flatten()

    return X_fft, X_dwt

def read_data(data_path):
    data_temp = (loadmat(data_path)["temp"]).flatten() 
    X_src = np.zeros(data_length, np.float64)
    if data_temp.shape[0]>data_length:
        X_src = data_temp[:data_length]
    else:
        X_src[0: data_temp.shape[0]] = data_temp
    return X_src

def read_dataPath(data_path):
    X = [] 
    labels = [] 
    indexes = []
    for i, data_folder in enumerate(os.listdir(data_path)):
        print("读取第{}个数据路径...............".format(i))
        for data in os.listdir(data_path +"/" + data_folder):
            X_path_one = data_path + "/" + data_folder + "/" + data

            index_one = data[data.find("_")+1 : data.find(".")]

            if "fear" in data_folder:
                labels_one = "fear"
            elif "joy" in data_folder:
                labels_one = "joy"

            X.append(X_path_one)
            labels.append(labels_one)
            indexes.append(index_one)
    return X, labels, indexes



def read_X(X):
    X_src = np.zeros((len(X), data_length), np.float64)
    for i in range(len(X)):
        print("加入第{}个个体数据...............".format(i))
        X_src[i] = read_data(X[i])
    return X_src


X, labels, indexes = read_dataPath(data_path)
X_src = read_X(X)
print(X_src.shape)

X_fft = myfft(X_src)
X_dwt = mydwt(X_src)
X = np.hstack((X_fft, X_dwt))

X_pca = model_pca.transform(X)
ss = StandardScaler()
X_ss = ss.fit_transform(X_pca)

for i in range(len(X_ss)):
    data = X_ss[i]
    label = labels[i]
    index = indexes[i]
    name = out_path + "/" + label + "_" + str(index) + "_frequency.mat"
    sio.savemat(name, {"data": data})


# ind = 6
# plt.figure()
# plt.subplot(231), plt.plot(X_src[ind]), plt.title("X_sec")
# plt.subplot(232), plt.plot(X_fft[ind]), plt.title("X_fft")
# plt.subplot(233), plt.plot(X_dwt[ind]), plt.title("X_dwt")
# plt.subplot(234), plt.plot(X[ind]),     plt.title("X")
# plt.subplot(235), plt.plot(X_pca[ind]), plt.title("X_pca")
# plt.subplot(236), plt.plot(X_ss[ind]),  plt.title("X_ss")
# plt.show()

# classify = "joy"
# index = 50
# data_path = out_path + "/{}_{}_frequency.mat".format(classify, index)
# a = (loadmat(data_path)["data"]).flatten() 
# plt.figure()
# plt.plot(a), plt.title("查看 {}_{}_frequency.mat".format(classify, index))
# plt.show()
