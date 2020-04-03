import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # Input to hidden function
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Input to output function
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # Softmax is only for categorical outputs. Ours is scalar output
        # self.softmax = nn.LogSoftmax(dim=1)

    # Called on each input
    # Computes the outputs (and next hidden state)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # Could use torch.rand also
        return torch.zeros(1, self.hidden_size).to(torch.device("cuda"))
