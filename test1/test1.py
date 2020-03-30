import sys
import os
import functools
import collections
import zlib
import string
import pickle

import parse

def get_usefulness():
    print('getting usefulness')
    with open('E_conj/statistics', 'r') as f:
        s = f.read()
        ls = s.split('\n')
        usefulness = collections.defaultdict(dict)
        for l in ls:
            if not l.strip():
                continue
            psr, problem, lemmaname, *_ = l.split(':')
            psr = float(psr)
            lemmaname = lemmaname.split('.')[0]
            usefulness[problem][lemmaname] = psr

    return usefulness

@functools.lru_cache(maxsize=1)
def parse_problem(problemname):
    return parse.parse_cnf_file('E_conj/problems/{}'.format(problemname))

def _process_problemslemmas(l):
    name, lemma = l.split(':')
    _, problemname, lemmaname = name.split('/')
    return (
        problemname,
        lemmaname,
        parse_problem(problemname),
        lemma,
        )

def get_problemslemmas():
    print('parsing problems and lemmas')
    import multiprocessing

    with multiprocessing.Pool() as pool:
        with open('E_conj/lemmas') as f:
            return pool.map(_process_problemslemmas, f, 32)

def shuffle_by_hash(l, key=str):
    return sorted(l,
            key=lambda x:
                zlib.crc32(key(x).encode('utf-8')) & 0xffffffff)

usefulness = None
problemlemmas = None
all_letters = None
n_letters = None

if (os.path.isfile('test1data/usefulness_raw.pickle')):
    usefulness = pickle.load(open('test1data/usefulness_raw.pickle', 'rb'))
else:
    usefulness = get_usefulness()
    pickle.dump(usefulness, open('test1data/usefulness_raw.pickle', 'wb'))

if (os.path.isfile('test1data/problemslemmas_raw.pickle')):
    problemlemmas = pickle.load(open('test1data/problemslemmas_raw.pickle', 'rb'))
else:
    problemlemmas = get_problemslemmas()
    pickle.dump(problemlemmas, open('test1data/problemslemmas_raw.pickle', 'wb'))

all_letters = string.printable
n_letters = len(all_letters)

import random
import torch
import torch.nn as nn

cuda = torch.device("cuda")

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor.to(cuda)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor.to(cuda)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # Input to hidden function
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Input to output function
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
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
    pl_probcatlemma = pl[2] + '@' + pl[3]
    pl_usefulness = usefulness[pl_probname][pl_lemmaname]

    usefulness_tensor = torch.tensor([[pl_usefulness]], dtype=torch.float).to(cuda)
    line_tensor = lineToTensor(pl_probcatlemma)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor

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

import time
import math

n_iters = 10000
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

print(all_losses)
