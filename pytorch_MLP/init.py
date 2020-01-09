class InitParser(object):
    def __init__(self):
        #gpu setting
        self.gpu_id = 0

        #training setting
        self.epoch = 300

        #path setting
        self.output_path = "../output"
        self.txt_path = "../txt/stack_feature"

        # hyper parameter
        self.batch_size = 100
        self.lr = 3e-5

        self.input_dimension = 1050  #输入数据的特征维度
        self.hidden_size = 256  #隐藏层1的维度
        self.num_class = 2  #类别数



