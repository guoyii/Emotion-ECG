import os
import re
from numpy import random
# path = '../MITBIH_ECG_preprocessing/MITBIH_ECG_preprocessing'
# output_path = '../MITBIH_ECG_preprocessing'

# data = []

# for root, dirs, files in os.walk(path):
#     for name in files:
#         label = re.search("_[0-9]*",name).group()
#         label = label[1:]
#         if label == '1':  #正常和非正常
#             data.append([os.path.join(root, name),'1'])
#         else:
#             data.append([os.path.join(root, name),'0'])

# random.shuffle(data) #打乱顺序

# train_file=open(os.path.join(output_path, "train.txt"), "a")
# val_file=open(os.path.join(output_path, "val.txt"), "a")
# test_file=open(os.path.join(output_path, "test.txt"), "a")

# l = len(data)

# #train
# for sequence in data[0:int(0.6*l)]:
#     train_file.write("{} {}\n".format(sequence[0],sequence[1]))
# #val
# for sequence in data[int(0.6*l):int(0.8*l)]:
#     val_file.write("{} {}\n".format(sequence[0],sequence[1]))
# #test
# for sequence in data[int(0.8*l):]:
#     test_file.write("{} {}\n".format(sequence[0],sequence[1]))

path = '../collected_data'
output_path = '../collected_Data'

data = []

for root, dirs, files in os.walk(path):
    for name in files:
        if name[0:3] == 'joy':  #正常和非正常
            data.append([os.path.join(root, name),'1'])
        else:
            data.append([os.path.join(root, name),'0'])

random.shuffle(data) #打乱顺序

train_file=open(os.path.join(output_path, "train.txt"), "a")
val_file=open(os.path.join(output_path, "val.txt"), "a")
test_file=open(os.path.join(output_path, "test.txt"), "a")

l = len(data)

#train
for sequence in data[0:int(0.6*l)]:
    train_file.write("{} {}\n".format(sequence[0],sequence[1]))
#val
for sequence in data[int(0.6*l):int(0.8*l)]:
    val_file.write("{} {}\n".format(sequence[0],sequence[1]))
#test
for sequence in data[int(0.8*l):]:
    test_file.write("{} {}\n".format(sequence[0],sequence[1]))