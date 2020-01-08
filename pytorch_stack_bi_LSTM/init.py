


class InitParser(object):
    def __init__(self):
        #gpu setting
        self.gpu_id = 2

        #training setting
        self.num_epoch = 200
        self.fine_tuning_epoch = 500
        self.is_fine_tuning = True

        #path setting
        self.output_path = "../output"
        self.data_path = ""
        self.txt_path = "../MITBIH_ECG_preprocessing"
        self.fine_tuning_txt_path = "../collected_data"


        # hyper parameter
        self.batch_size = 1000
        self.fine_tuning_batch_size = 10
        self.lr = 3e-5

        self.num_layers = 3
        # self.seq_len = 300
        self.input_size = 1
        self.hidden_size = 20
        self.fc_hidden_size = 1000
        self.num_class = 2



