import math
import random
import string
import sys
import time
import torch
import torch.nn as nn
from util import *
from parse import *

cuda = torch.device("cuda")

usefulness, problemlemmas = get_usefulness_problemslemmas()
all_letters = string.printable
n_letters = len(all_letters)

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
        return torch.zeros(1, self.hidden_size).to(cuda)

# Output is just a float, the proof length ratios
output_size = 1
n_hidden = 128

rnn = RNN(n_letters, n_hidden, output_size)
rnn.to(cuda)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    pl = randomChoice(problemlemmas)
    pl_probname = pl[0]
    pl_lemmaname = pl[1]
    # Concatenate the problem with lemma, seperated by unused '@'
    pl_probcatlemma = pl[2] + '@' + pl[3]
    pl_usefulness = usefulness[pl_probname][pl_lemmaname]

    usefulness_tensor = torch.tensor([[pl_usefulness]], dtype=torch.float).to(cuda)
    line_tensor = lineToTensor(pl_probcatlemma, all_letters, cuda)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor

# The loss function is mean squared, this is different from the tutorial
criterion = nn.MSELoss()
learning_rate = 0.005

def train(usefulness_tensor, line_tensor):
    hidden = rnn.initHidden()
    output = None

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, usefulness_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

n_iters = 10
print_every = 10
plot_every = 10

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    pl_probname, pl_lemmaname, usefulness_tensor, line_tensor = randomTrainingExample()
    output, loss = train(usefulness_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print('\nIteration: %d \tProgress: %d%% \t(%s)' % (iter, iter / n_iters * 100, timeSince(start)))
        print('Loss: %.4f \tTarget: %s \tOutput: %s' % (loss, usefulness_tensor.data[0][0], output.data[0][0]))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # Sanity check that everything is still running
    sys.stdout.write('#')
    sys.stdout.flush()

torch.save(rnn.state_dict(), './test1models/training.pt')

print(all_losses)
