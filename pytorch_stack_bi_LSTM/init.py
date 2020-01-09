class InitParser(object):
    def __init__(self):
        #gpu setting
        self.gpu_id = 0

        #training setting
        self.pretain_epoch = 200
        self.fine_tuning_epoch = 500
        self.is_fine_tuning = True

        #path setting
        self.output_path = "../output"
        self.pretain_txt_path = "../txt/MITBIH_ECG_preprocessing"
        self.fine_tuning_txt_path = "../txt/collected_data_full"

        # hyper parameter
        self.pretain_batch_size = 1000
        self.fine_tuning_batch_size = 10
        self.lr = 3e-5

        self.num_layers = 3
        self.input_size = 1  #输入数据的特征维度
        self.hidden_size = 20  #LSTM隐藏层的特征维度
        self.fc_hidden_size = 1000
        self.num_class = 2



