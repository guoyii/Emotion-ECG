from torch import nn

class MLP(nn.Module):
    def __init__(self,parsers):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = parsers.input_dimension,out_features = parsers.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = parsers.hidden_size, out_features = parsers.num_class)


    def forward(self,x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1,2)