import math
import os
import pdb
import string
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
random.seed(time.time())

from cuda_check import device
from parse import *
from lstm import LSTM
from util import *

usefulness, problemlemmas_test, problemslemmas_validation = get_data()
random.shuffle(problemlemmas_test)

# Output is float, sigmoid function output
input_size = 2 * tensor_length
output_size = 1
n_hidden1 = 128
n_hidden2 = 32

lstm = LSTM(input_size, n_hidden1, n_hidden2, output_size)
lstm.to(device)

# The loss function is binary cross entropy
criterion = nn.BCELoss()
learning_rate = 0.005
optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)

def train(usefulness_tensor, line_tensor):
    output = None

    lstm.zero_grad()

    output = lstm(line_tensor)
    output = output[output.size()[0]-1]

    loss = criterion(output, usefulness_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

n_iters = 1000 # was 1000, made it 10 for now to stay consistent with numbers in test1.py
print_every = 10 # was 20, made it 10 for now to stay consistent with numbers in test1.py
plot_every = 10

n_validate = 25
validate_every = 200

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

# use this to perform validations halfway through the training (to see how well it's learning)
# not going to shuffle (think it makes more sense this way)
def _perform_midway_validation(model, validation_set, n_validate):
    
    correct = 0

    for i in range(n_validate):
        _,_, usefulness_tensor, line_tensor = getTrainingExample(
            validation_set[i], usefulness, device
        )

        output = lstm(line_tensor)
        output = output[output.size()[0]-1]
        o = output[0][0].item()
        t = usefulness_tensor[0][0].item()

        if (abs(o-t) < 0.5):
            correct += 1

    return (correct, n_validate, correct*100/n_validate)

for iter in range(1, n_iters + 1):
    pl_probname, pl_lemmaname, usefulness_tensor, line_tensor = getTrainingExample(
        problemlemmas_test[iter], usefulness, device
    )
    output, loss = train(usefulness_tensor, line_tensor)
    current_loss += loss
    total_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print('\nIteration: %d \tProgress: %d%% \t(%s)' % (iter, iter / n_iters * 100, timeSince(start)))
        print('Loss: %.4f \tTarget: %s \tOutput: %s' % (loss, usefulness_tensor.data[0][0], output.data[0][0]))
        print('Average Loss (total): %.4f' % (total_loss / iter))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    
    # Validate on a small validation set (same set used throughout)
    if iter % validate_every == 0:
        print('Validation: %d/%d (%d%%)' % _perform_midway_validation(lstm, problemslemmas_validation, n_validate))

    # Sanity check that everything is still running
    sys.stdout.write('#')
    sys.stdout.flush()

filename = './test4models/training.pt'
if not os.path.exists(os.path.dirname(filename)):
    os.makedirs(os.path.dirname(filename))

torch.save(lstm.state_dict(), filename)

print(all_losses)
