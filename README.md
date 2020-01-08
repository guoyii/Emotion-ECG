Frequency
=====  
  `main_self.py`:  Use the ecg data collected by ourselves to find the best dimension for classification  
  `main_mit.py`: The MIT dataset was used to find the best combination of frequency-domain features suitable for classification  
  `ShowResult.py`: Show some results  
  `test.py`: Generate the data we want  
  `utils.py`: Some functions used in `main_self.py`、`main_mit.py`...  
  `text.txt`: Experimental results looking for the best combination and the best dimension  

Preprocess
===== 
  `rrinterval_seg.m`: MIT_BIH dataset segment, based on label_time saved in ATRTIME  
  `Q1.m`: Our ecg data preprocess, denoise is not included, which is done via Matlab 1-D Wavelet toolbox.  
  `rdata.m`: MIT_BIH dataset preprocess, including how to read the file of 'hea, art...', from Machine_Learning_ECG-master.  

Result
=====  
  ***Data_H5***: FFT data、 DWT data and so on.  
  ***model_LinearSVC***: LinearSVC models trained with FFT and DWT data to find the best dimension.  
  ***model_pca***: PCA  models trained with FFT and DWT data which is used as a dimension reduction.  
  ***out***: Frequency domain characteristics of ecg signals.  