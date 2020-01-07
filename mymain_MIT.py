from sklearn.decomposition import PCA  ## 用于PCA降维，需要一起训练
from sklearn.externals import joblib   ## 用于保存和加载已经训练好的PCA模型
from sklearn.svm import LinearSVC      ##
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
                                       ## 标准化处理
from utils import read_dataPath
from utils import read_data
from utils import myfft
from utils import mydwt
from utils import mywavedec
import numpy as np
import h5py
import time
import os

print("\n******************   加载路径   ******************")
result_path = "V:/users/gy/MyProject/ECG/result"                       ## 存放结果目录
data_path = "V:/users/gy/data/ECG/MITBIH_ECG"                          ## 存放数据根目录
data_folder = "sample"                                                 ## 各样本文件夹名称 
n_components = 500
model_pca_name = "model_pca_fft+dwt_"
model_Lsvc_name = "model_lsvc_fft+dwt_"


if not os.path.isdir(result_path + "/Data_H5"):              ## 在result_path目录下生成data_h5文件夹，用于存放数据
    os.makedirs(result_path + "/Data_H5")
data_h5_path = result_path + "/Data_H5" 



print("\n******************   加载数据   ******************")
## 读取数据，将数据存于X，label存入变量labels 将其保存为h5格式
##********************************************************************************************************
if os.path.exists(data_h5_path + "/data_src.h5"):            ## 已经存在数据，则直接加载
    print("加载数据中...............")
    o = h5py.File(data_h5_path + "/data_src.h5", 'r')
    X_src = np.array(o['X'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    print("生成数据中...............")
    X_src, labels = read_dataPath(data_path)                     ## 无数据，则读入数据并保存
    X_src, labels = read_data(X_src, labels) 
    f = h5py.File(data_h5_path + "/data_src.h5", 'w')
    f.create_dataset('X', data=X_src)
    f.create_dataset('labels', data=labels)
    f.close()
print("加载数据完成，共{}个样本！".format(len(X_src)))




data_name = "data_fft.h5"
print("\n******************   傅里叶变换   ******************")
## 对数据进行傅里叶变换
##********************************************************************************************************
if os.path.exists(data_h5_path + "/" +data_name):            ## 已经存在数据，则直接加载
    print("加载傅里叶数据中...............")
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_fft = np.array(o['X_fft'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    print("生成傅里叶数据中...............")
    X_fft = myfft(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_fft', data=X_fft)
    f.create_dataset('labels', data=labels)
    f.close()
print("傅里叶变换完成! shape:{}".format(X_fft.shape))


data_name = "data_dwt.h5"
print("\n******************   小波变换   ******************")
## 对数据进行小波变换
##********************************************************************************************************
if os.path.exists(data_h5_path + "/" +data_name):            ## 已经存在数据，则直接加载
    print("加载小波数据中...............")
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_dwt = np.array(o['X_dwt'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    print("生成小波数据中...............")
    X_dwt = mydwt(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_dwt', data=X_dwt)
    f.create_dataset('labels', data=labels)
    f.close()
print("小波变换完成! shape:{}".format(X_dwt.shape))


data_name = "data_wavedec.h5"
print("\n******************   小波分解   ******************")
## 对数据进行小波分解
##********************************************************************************************************
if os.path.exists(data_h5_path + "/" +data_name):            ## 已经存在数据，则直接加载
    print("加载分解数据中...............")
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_wavedec = np.array(o['X_wavedec'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    print("生成分解数据中...............")
    X_wavedec = mywavedec(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_wavedec', data=X_wavedec)
    f.create_dataset('labels', data=labels)
    f.close()
print("小波分解完成! shape:{}".format(X_wavedec.shape))

##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
X = np.hstack((X_fft, X_dwt))
print("X.shape:", X.shape)

##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

print("\n******************   加载PCA模型   ******************")
## 加载或生成PCA降维模型并训练
##********************************************************************************************************
if not os.path.isdir(result_path + "/model_pca"):                                               ## 在result_path目录下生成model_pca文件夹，用于存放PCA模型
    os.makedirs(result_path + "/model_pca")
model_pca_path = result_path + "/model_pca"


if os.path.exists(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl"):          # 已经存在模型，则直接加载
    print("加载PCA...............")
    model_pca = joblib.load(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl") # 加载PCA模型
else:
    start = time.clock()
    print("生成PCA并训练模型...............")
    model_pca =PCA(n_components=n_components)                                                  # 创建模型                         
    model_pca.fit(X)                                                                           # 训练模型
    joblib.dump(model_pca, model_pca_path + "/" + model_pca_name + str(n_components) +".pkl")  # 保存PCA模型
    print("保存模型：{}".format(model_pca_name + str(n_components) +".pkl"))
    print("Time: %g s  <==>  %g min"%(time.clock() - start, (time.clock() - start)/60)) 
print("成功加载模型：{}".format(model_pca_name + str(n_components) +".pkl"))    


print("\n******************   对数据进行降维并标准化处理   ******************")
## 对数据进行降维并标准化
##********************************************************************************************************
X_pca = model_pca.transform(X)                                                                # 对数据进行降维
# print(X_pca.explained_variance_ratio_)                                                      # 输出贡献率：不知道具体表示啥，可以选择查看

ss = StandardScaler()
X = ss.fit_transform(X_pca)
print("降维完成！")
print("数据集 shape: ", X.shape)                                                               # 显示降维后数据维度是否正确


print("\n******************  划分为训练集与验证集  ******************")
## 将数据划分为训练集与验证级
##********************************************************************************************************
X_train, X_test, label_train, label_test = train_test_split(X, labels, test_size = 0.2, random_state = 0)
print("训练集： data.shape:{}  label.shape:{}".format(X_train.shape, label_train.shape))
print("测试集： data.shape:{}  label.shape:{}".format(X_test.shape, label_test.shape))



print("\n******************   加载SVC模型   ******************")
## 加载或生成SVC降维模型并训练
##********************************************************************************************************
if not os.path.isdir(result_path + "/model_LinearSVC"):                                          ## 在result_path目录下生成model_LinearSVC文件夹，用于存放LinearSVC模型
    os.makedirs(result_path + "/model_LinearSVC")
model_Lsvc_path = result_path + "/model_LinearSVC"


if os.path.exists(model_Lsvc_path + "/" +model_Lsvc_name + str(n_components) +".pkl"):           # 已经存在模型，则直接加载
    print("加载SVC...............")
    model_lsvc = joblib.load(model_Lsvc_path + "/" +model_Lsvc_name + str(n_components) +".pkl") # 加载PCA模型
else:
    t0 = time.clock()
    print("生成SVC并训练模型...............") 
    model_lsvc = LinearSVC()                                                                     # 创建模型                         
    model_lsvc.fit(X_train, label_train)                                                         # 训练模型
    joblib.dump(model_lsvc, model_Lsvc_path + "/" + model_Lsvc_name + str(n_components) +".pkl") # 保存PCA模型
    print("保存模型：{}".format(model_Lsvc_name + str(n_components) +".pkl"))
    print("Time for training the LinearSVC: %g s  <==>  %g min"%(time.clock() - t0, (time.clock() - t0)/60))   
print("成功加载模型：{}".format(model_Lsvc_name + str(n_components) +".pkl")) 


## 显示分类结果
##********************************************************************************************************
y_predict = model_lsvc.predict(X_test)
print('The Accuracy of LinearSVC is:', model_lsvc.score(X_test, label_test))
