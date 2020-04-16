import torch
import torch.nn as nn

from cuda_check import device  

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.hidden2out(lstm_out)
        output = self.sigmoid(output)
        return output
