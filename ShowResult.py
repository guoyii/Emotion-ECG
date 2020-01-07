from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import h5py
data_h5_path = "V:/users/gy/MyProject/ECG/result/Data_H5"


data_origin = h5py.File(data_h5_path + "/data_src.h5", 'r')
X_origin = np.array(data_origin['X'][:])
labels_origin = np.array(data_origin['labels'][:])


data_fft = h5py.File(data_h5_path + "/data_fft.h5", 'r')
X_fft = np.array(data_fft['X'][:])
labels_fft = np.array(data_fft['labels'][:])

plt.figure()
plt.subplot(321), plt.plot(range(X_origin[0].shape[0]), X_origin[0]), plt.xlim(0, 800), plt.title("X_origin[0]")
plt.subplot(322), plt.plot(range(X_fft[0].shape[0]), X_fft[0]),       plt.xlim(0, 800), plt.title("X_fft[0]")
plt.subplot(323), plt.plot(range(X_origin[1].shape[0]), X_origin[0]), plt.xlim(0, 800), plt.title("X_origin[1]")
plt.subplot(324), plt.plot(range(X_fft[1].shape[0]), X_fft[0]),       plt.xlim(0, 800), plt.title("X_fft[1]")
plt.subplot(325), plt.plot(range(X_origin[2].shape[0]), X_origin[0]), plt.xlim(0, 800), plt.title("X_origin[2]")
plt.subplot(326), plt.plot(range(X_fft[2].shape[0]), X_fft[0]),       plt.xlim(0, 800), plt.title("X_fft[2]")
plt.show()

