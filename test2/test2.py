import math
import os
import string
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from cuda_check import device
from parse import *
from lstm import LSTM
from util import *

usefulness, problemlemmas = get_usefulness_problemslemmas()
all_letters = string.printable
n_letters = len(all_letters)

# Output is just a float, the proof length ratios
output_size = 1
n_hidden = 128

lstm = LSTM(n_letters, n_hidden, output_size)
lstm.to(device)

# The loss function is mean squared, this is different from the tutorial
criterion = nn.MSELoss()
learning_rate = 0.005
optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)

def train(usefulness_tensor, line_tensor):
    output = None

    lstm.zero_grad()

    # am unsure if we are even passing in the right dims
    output = lstm(line_tensor)
    output = output[output.size()[0]-1]
    
    loss = criterion(output, usefulness_tensor)
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

n_iters = 1000
print_every = 20
plot_every = 10

# Keep track of losses for plotting
current_loss = 0
total_loss = 0
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
    total_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print('\nIteration: %d \tProgress: %d%% \t(%s)' % (iter, iter / n_iters * 100, timeSince(start)))
        print('Loss: %.4f \tTarget: %s \tOutput: %s' % (loss, usefulness_tensor.data[0][0], output.data[0][0]))
        print('Average Loss: %.4f' % (total_loss / iter))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # Sanity check that everything is still running
    sys.stdout.write('#')
    sys.stdout.flush()

filename = './test2models/traning.pt'
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

torch.save(lstm.state_dict(), filename)

print(all_losses)
