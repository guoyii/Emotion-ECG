clc; clear all;close all
string1='ecg_signal_';
string2='label_';
for count=1:48 %% 48 samples
   temp_sample=strcat(string1, num2str(count));
   temp_sample_label=strcat(string2,num2str(count));
%    nf=strcat('/home/lq/ECG/Machine_Learni,'sample');
   new_folder=strcat('sample',num2str(count));
   mkdir(new_folder)
   path=strcat('/home/lq/ECG/Machine_Learning_ECG-master/',new_folder);
   path=strcat(path,'/');
% save([path, 'test1'], 'count')
load(temp_sample)
load( temp_sample_label)

%label_time=[label_time label_time(:,2)*360]; % column 3 stands for label num
label_time=[c c(:,2)*360]; % sample rate: 360
ecg=(eval(temp_sample));
ecg1=ecg(:,1);
% for j=1%%samples total 48%
  for i=1:length(label_time(:,1))-1 % rr interval 
% for i=1:4
    temp=ecg1(label_time(i,3):label_time(i+1,3));
    str='_ecg';
    b=strcat(num2str(count),'_');
    b1=strcat(b, num2str(label_time(i,1)));
    b=strcat(b1,str);
    b=strcat(b,num2str(i));
   
%     c=str2sym(b);
    save([path, b],'temp')
end
end