Guide
=====  
  `main_SELF.py`: Use the ecg data collected by ourselves to find the best dimension for classification  
  `main_MIT.py`:The MIT dataset was used to find the best combination of frequency-domain features suitable for classification  
  `ShowResult.py`:Show some results  
  `test.py`:Generate the data we want  
  `utils.py`:Partial functions used  
  `TEXT.txt`:Experimental results

Result
=====  
Data_H5
______  
  A folder where data is stored   

model_LinearSVC
______
  A folder for LinearSVC  

model_pca
______  
  A folder for PCA  

data_preprocessing
===== 
  `rrinterval_seg.m`: MIT_BIH dataset segment, based on label_time saved in ATRTIME 
  
  `Q1.m`: our ecg data preprocess, denoise is not included, which is done via Matlab 1-D Wavelet toolbox
