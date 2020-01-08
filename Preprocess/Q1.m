%version 1 detect RR INTERVAL 
% modified based on max
% detect ECG OF fear and joy 
% LQ 1223
clear all; close all; clc
load('joy4.txt')
load('joy2_dn.mat')
x=joy4(1:end);

fear=x((285*125+1):315*125);
joy=x((705*125+1):735*125);

x=joy(1:1500);
figure
plot(x)
title('ECG signal before filtering')
ylabel('Amplitude')   %plotting the raw ECG signal data
% print('Before_Filter.jpg','-djpeg')
% x=fear;
 %especially for 007
w = 50/(256/2); %256
bw = w;
[num, den] = iirnotch(w,bw);
x_notch = filter(num, den, x);
figure
plot ( x_notch, 'r')
title(' ECG signal after notch filtering')
ylabel ('Amplitude')  %plotting filtered ECG
% print('After_Notch_Filter.jpg','-djpeg')


ecg_wave = df(x_notch);
figure
plot( ecg_wave)
title('ECG signal after diffrentiation')
ylabel ('Amplitude')  %plotting ECG after diff
% print('After_Diff.jpg','-djpeg')

%%ecg_sqrd = ecg_wave.^2;
ecg_sqrd=ecg_wave.^2;

figure
plot(ecg_sqrd, 'r');
title('Squared ECG signal') %squared signal
ylabel ('Amplitude')  %plotting ECG after squaring
% print('After_Squaring.jpg','-djpeg')
% ecg_sqrd=ecg_sqrd(80:end);%?
N = 7; %for 7 fear
ecg_smooth = hsmooth(ecg_sqrd, N);
id = peak(ecg_smooth, 0.6*max(ecg_smooth));%%0.6
figure, hold on
plot(ecg_smooth)
scatter(id, ecg_smooth(id));
title('Detected RR interval');

hold off
% print(sprintf('DetectedR_%d.jpg',N),'-djpeg')

figure,
plot((id*1000)/256);
title('RR');
xlabel('beats number')
ylabel('RR interval');
% fear1_dn=(fear1_dn-mean(fear1_dn))./std(fear1_dn); %normalize
joy2_dn=(joy2_dn-mean(joy2_dn))./std(joy2_dn);
%% rr interval segment by LQ 1223
id_minus= id-4; % 6 is just for test
% new_folder1='joy';
% mkdir(new_folder1)
path=strcat('C:\Users\Admin\Desktop\ECG_course\QRS-detection-method-master\','joy');
path=strcat(path,'\');
str='joy_';
% for i=1:2
for i=1:length(id)-1
% for i=1:10
    temp=joy2_dn((id_minus(i)+1):(id_minus(i+1)-1));
    temp=interp(temp,3);% upsampling for a interger ,here I chose 3
    temp=temp';
    str1=strcat(str,num2str(i+13));
    save([path str1],'temp')
end
