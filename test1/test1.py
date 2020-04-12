import math
import os
import string
import sys
import time
import torch
import torch.nn as nn

from cuda_check import device
from parse import *
from rnn import RNN
from util import *

usefulness, problemlemmas = get_usefulness_problemslemmas()
all_letters = string.printable
n_letters = len(all_letters)

# Output is just a float, the proof length ratios
output_size = 1
n_hidden = 128

rnn = RNN(n_letters, n_hidden, output_size)
rnn.to(device)

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
    pl_probname, pl_lemmaname, usefulness_tensor, line_tensor = randomTrainingExample(
        problemlemmas, usefulness, all_letters, device
    )
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

filename = './test1models/training.pt'
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

torch.save(rnn.state_dict(), filename)

print(all_losses)
