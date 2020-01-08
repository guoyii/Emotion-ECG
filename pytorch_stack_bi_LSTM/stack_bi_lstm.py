from torch import nn

class Stack_Bi_LSTM(nn.Module):
    def __init__(self,parsers):
        super(Stack_Bi_LSTM, self).__init__()
        self.num_layers = parsers.num_layers
        self.num_directions = 2
        self.batch_size = parsers.batch_size
        self.hidden_size = parsers.hidden_size
        self.lstm = nn.LSTM(input_size = parsers.input_size,hidden_size=parsers.hidden_size,num_layers=parsers.num_layers,bidirectional=True)
        self.fc1 = nn.Linear(in_features = parsers.hidden_size,out_features = parsers.fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = parsers.fc_hidden_size, out_features = parsers.num_class)


    def forward(self,x):
        _,(x,_) = self.lstm(x)
        x = self.fc1(x.view(self.num_layers, self.num_directions, x.shape[1], self.hidden_size)[-1,:,:,:].sum(0))
        x = self.relu(x)
        x = self.fc2(x)
        return x