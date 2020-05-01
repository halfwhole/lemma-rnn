import torch
import torch.nn as nn

from cuda_check import device  

# stacked lstm with 2 hidden layers that can be of diff sizes
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LSTM, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lstm1 = nn.LSTM(input_size, hidden_size1)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2)
        self.hidden2out = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        lstm_out1, _ = self.lstm1(input)
        lstm_out2, _ = self.lstm2(lstm_out1)
        output = self.hidden2out(lstm_out2)
        output = self.sigmoid(output)
        return output
