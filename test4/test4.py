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

filename = './test4models/training.pt'

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

def trainOne(usefulness_tensor, line_tensor):
    output = None

    lstm.zero_grad()

    output = lstm(line_tensor)
    output = output[output.size()[0]-1]

    loss = criterion(output, usefulness_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

n_iters = 10000
train_every = 10 # was 20, made it 10 for now to stay consistent with numbers in test1.py

n_validate = 10
validate_every = 10

# Keep track of losses for plotting
all_losses = []
all_test_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# use this to perform validations halfway through the training (to see how well it's learning)
def test(model, validation_set, n_validate):

    random.shuffle(validation_set)
    total_loss = 0

    for i in range(n_validate):
        _,_, usefulness_tensor, line_tensor = getTrainingExample(
            validation_set[i], usefulness, device
        )

        output = lstm(line_tensor)
        output = output[output.size()[0]-1]
        loss = criterion(output, usefulness_tensor).item()
        total_loss += loss

    return total_loss / n_validate

def train(n_iters):

    start = time.time()

    global all_losses, all_test_losses
    current_loss = 0
    total_loss = 0

    for iter in range(1, n_iters + 1):
        pl_probname, pl_lemmaname, usefulness_tensor, line_tensor = getTrainingExample(
            problemlemmas_test[iter], usefulness, device
        )
        output, loss = trainOne(usefulness_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        # Add current loss avg to list of losses
        if iter % train_every == 0:
            print('\nIteration: %d \tProgress: %d%% \t(%s)' % (iter, iter / n_iters * 100, timeSince(start)))
            print('Loss: %.4f \tTarget: %s \tOutput: %s' % (loss, usefulness_tensor.data[0][0], output.data[0][0]))
            print('Average Loss: %.4f' % (current_loss / train_every))
            all_losses.append(current_loss / train_every)
            current_loss = 0

        # Validate on a small validation set
        # if iter % validate_every == 0:
        #     average_test_loss = test(lstm, problemslemmas_validation, n_validate)
        #     print('Average Test Loss (over %d test examples): %.4f' % (n_validate, average_test_loss))
        #     all_test_losses.append(average_test_loss)

        # Sanity check that everything is still running
        sys.stdout.write('#')
        sys.stdout.flush()

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    torch.save(lstm.state_dict(), filename)
    print('Saved model to %s!' % filename)

    print('All Losses:')
    print(all_losses)

# lstm.load_state_dict(torch.load(filename))
# lstm.eval()

#################################### Validation ####################################

n_iters_validate = 1000
print_every_validate = 10
print_every_correct = 0

def validate(n_iters_validate):

    global print_every_validate, print_every_correct
    total_correct = 0
    total_loss = 0
    start_validate = time.time()

    # Note: This is not midway-validation. This is validation after training on dataset.
    for iter in range(1, n_iters_validate + 1):
        _,_, usefulness_tensor, line_tensor = getTrainingExample(
        problemslemmas_validation[iter - 1], usefulness, device
        )

        output = lstm(line_tensor)
        output = output[output.size()[0]-1]
        o = output[0][0].item()
        t = usefulness_tensor[0][0].item()
        loss = criterion(output, usefulness_tensor).item()
        total_loss += loss

        if (abs(o-t) < 0.5):
            print_every_correct += 1
            total_correct += 1

        # Sanity check that everything is still running
        sys.stdout.write('#')
        sys.stdout.flush()

        # Print iter number, and number correctly classified
        if iter % print_every_validate == 0:
            print('\nIteration: %d \tProgress: %d%% \t(%s)' % (iter, iter / n_iters_validate * 100, timeSince(start_validate)))
            print('Validation: %d/%d (%d%%)' % (print_every_correct, print_every_validate, print_every_correct*100/print_every_validate))
            print_every_correct = 0

    print('Final Validation Results: %d/%d (%d%%)' % (total_correct, n_iters_validate, total_correct*100/n_iters_validate))
    print('Average Validation Loss: %.4f' % (total_loss / n_iters_validate))

train(n_iters)
validate(n_iters_validate)
