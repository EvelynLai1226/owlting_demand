import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 15)
        self.fc2 = nn.Linear(15+6*180, 6*180)
        self.sigmoid = nn.Sigmoid()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, input_2, hidden=None):
        x1, (hn, cn) = self.lstm(input, hidden)
        x1 = x1[:, -1]  # ht
        # x1 = hn
        x1 = self.fc1(x1)
        x1 = self.sigmoid(x1)

        x2 = input_2.flatten().view(input_2.shape[0], -1)
        # x3 = input_3
        # x4 = input_4
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output

    def initHidden(self, batch_size):
        hidden0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        cell0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (hidden0, cell0)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)