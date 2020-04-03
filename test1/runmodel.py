import random
import string
import sys
import torch
import torch.nn as nn
from parse import *
from rnn import RNN
from util import *

cuda = torch.device("cuda")

usefulness, problemlemmas = get_usefulness_problemslemmas()
all_letters = string.printable
n_letters = len(all_letters)

# Output is just a float, the proof length ratios
output_size = 1
n_hidden = 128

model = RNN(n_letters, n_hidden, output_size)
state_dict = torch.load('./test1models/training.pt')
model.load_state_dict(state_dict)
model.to(cuda)

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

criterion = nn.MSELoss()
def eval_model(usefulness_tensor, line_tensor):
    hidden = model.initHidden()
    output = None

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, usefulness_tensor)

    return output, loss.item()

output = None
while (1):
    print("Enter to test random data instance")
    input()
    pl_probname, pl_lemmaname, usefulness_tensor, line_tensor = randomTrainingExample()
    output, loss = eval_model(usefulness_tensor, line_tensor)
    print('Problem: %s \tLemma: %s' % (pl_probname, pl_lemmaname))
    print('Loss: %.4f \tTarget: %s \tOutput: %s' % (loss, usefulness_tensor.data[0][0], output.data[0][0]))

