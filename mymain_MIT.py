from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import read_dataPath
from utils import read_data
from utils import myfft
from utils import mydwt
from utils import mywavedec
import numpy as np
import h5py
import time
import os

result_path = "V:/users/gy/MyProject/ECG/result" 
data_path = "V:/users/gy/data/ECG/MITBIH_ECG"
data_folder = "sample"  
n_components = 500
model_pca_name = "model_pca_fft+dwt_"
model_Lsvc_name = "model_lsvc_fft+dwt_"


if not os.path.isdir(result_path + "/Data_H5"): 
    os.makedirs(result_path + "/Data_H5")
data_h5_path = result_path + "/Data_H5" 


if os.path.exists(data_h5_path + "/data_src.h5"): 
    o = h5py.File(data_h5_path + "/data_src.h5", 'r')
    X_src = np.array(o['X'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    X_src, labels = read_dataPath(data_path) 
    X_src, labels = read_data(X_src, labels) 
    f = h5py.File(data_h5_path + "/data_src.h5", 'w')
    f.create_dataset('X', data=X_src)
    f.create_dataset('labels', data=labels)
    f.close()


data_name = "data_fft.h5"
if os.path.exists(data_h5_path + "/" +data_name): 
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_fft = np.array(o['X_fft'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    X_fft = myfft(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_fft', data=X_fft)
    f.create_dataset('labels', data=labels)
    f.close()


data_name = "data_dwt.h5"
if os.path.exists(data_h5_path + "/" +data_name): 
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_dwt = np.array(o['X_dwt'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    X_dwt = mydwt(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_dwt', data=X_dwt)
    f.create_dataset('labels', data=labels)
    f.close()


data_name = "data_wavedec.h5"
if os.path.exists(data_h5_path + "/" +data_name): 
    o = h5py.File(data_h5_path + "/" +data_name, 'r')
    X_wavedec = np.array(o['X_wavedec'][:])
    labels = np.array(o['labels'][:])
    o.close()
else:
    X_wavedec = mywavedec(X_src[:])
    f = h5py.File(data_h5_path + "/" + data_name, 'w')
    f.create_dataset('X_wavedec', data=X_wavedec)
    f.create_dataset('labels', data=labels)
    f.close()

##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
X = np.hstack((X_fft, X_dwt))
print("X.shape:", X.shape)

##////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if not os.path.isdir(result_path + "/model_pca"):
    os.makedirs(result_path + "/model_pca")
model_pca_path = result_path + "/model_pca"


if os.path.exists(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl"): 
    model_pca = joblib.load(model_pca_path + "/" + model_pca_name + str(n_components) +".pkl") 
else:
    start = time.clock()
    model_pca =PCA(n_components=n_components)                        
    model_pca.fit(X) 
    joblib.dump(model_pca, model_pca_path + "/" + model_pca_name + str(n_components) +".pkl")
    print("Save model：{}".format(model_pca_name + str(n_components) +".pkl"))
    print("Time: %g s  <==>  %g min"%(time.clock() - start, (time.clock() - start)/60)) 
print("Success：{}".format(model_pca_name + str(n_components) +".pkl"))    

X_pca = model_pca.transform(X) 


ss = StandardScaler()
X = ss.fit_transform(X_pca)
print("Done！")
print("Dataset shape: ", X.shape)  

X_train, X_test, label_train, label_test = train_test_split(X, labels, test_size = 0.2, random_state = 0)
print("Train： data.shape:{}  label.shape:{}".format(X_train.shape, label_train.shape))
print("Test： data.shape:{}  label.shape:{}".format(X_test.shape, label_test.shape))


if not os.path.isdir(result_path + "/model_LinearSVC"):  
    os.makedirs(result_path + "/model_LinearSVC")
model_Lsvc_path = result_path + "/model_LinearSVC"


if os.path.exists(model_Lsvc_path + "/" +model_Lsvc_name + str(n_components) +".pkl"):  
    model_lsvc = joblib.load(model_Lsvc_path + "/" +model_Lsvc_name + str(n_components) +".pkl") 
else:
    t0 = time.clock()
    model_lsvc = LinearSVC()                                                                                        
    model_lsvc.fit(X_train, label_train)                                                       
    joblib.dump(model_lsvc, model_Lsvc_path + "/" + model_Lsvc_name + str(n_components) +".pkl")
    print("Save model：{}".format(model_Lsvc_name + str(n_components) +".pkl"))
    print("Time for training the LinearSVC: %g s  <==>  %g min"%(time.clock() - t0, (time.clock() - t0)/60))   
print("Success：{}".format(model_Lsvc_name + str(n_components) +".pkl")) 


y_predict = model_lsvc.predict(X_test)
print('The Accuracy of LinearSVC is:', model_lsvc.score(X_test, label_test))
