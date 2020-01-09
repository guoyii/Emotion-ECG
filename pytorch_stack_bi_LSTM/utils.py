import matplotlib.pyplot as plt
import os 

class AvgMeter(object):
	""" this class is to record one variable such as loss or acc """

	def __init__(self):
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.val = 0
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)




def plot(data1,data2, output_path, y_label1,y_label2,batch_size,name, color1='r',color2='b', x_label="epoch",pre_train = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data1, color1, label=y_label1)
    ax.plot(data2, color2, label=y_label2)
    if pre_train:
        ax.set_title("Pre-train with batch_size=%d"%(batch_size))
    else:
        ax.set_title("Fine-tuning with batch_size=%d"%(batch_size))
    ax.set_xlabel(x_label)
    ax.set_ylabel(name)
    plt.grid()  # 生成网格
    plt.legend()
    plt.savefig(os.path.join(output_path, "{}.png".format(name)))
    plt.close()