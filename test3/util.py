import pdb
import random
import torch

# Find letter index from all_letters, e.g. "a" = 0
def _letterToIndex(letter, all_letters):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, all_letters, cuda):
    n_letters = len(all_letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][_letterToIndex(letter, all_letters)] = 1
    return tensor.to(cuda)

def categoryIndexFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

def _randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# TODO refactor this to use the getTrainingExample method
def randomTrainingExample(problemlemmas, usefulness, all_letters, cuda):
    pl = _randomChoice(problemlemmas)
    pl_probname = pl[0]
    pl_lemmaname = pl[1]
    # Concatenate the problem with lemma, seperated by unused '@'
    pl_probcatlemma = pl[2] + '@' + pl[3]
    pl_usefulness = usefulness[pl_probname][pl_lemmaname]
    if (pl_usefulness < 1):
        usefulness_tensor = torch.tensor([[1]], dtype=torch.float).to(cuda)
    else:
        usefulness_tensor = torch.tensor([[0]], dtype=torch.float).to(cuda)

    usefulness_tensor = torch.tensor([[pl_usefulness]], dtype=torch.float).to(cuda)
    line_tensor = lineToTensor(pl_probcatlemma, all_letters, cuda)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor

def getTrainingExample(pl, usefulness, all_letters, cuda):
    pl_probname = pl[0]
    pl_lemmaname = pl[1]
    pl_usefulness = usefulness[pl_probname][pl_lemmaname]
    if (pl_usefulness < 1):
        usefulness_tensor = torch.tensor([[1]], dtype=torch.float).to(cuda)
    else:
        usefulness_tensor = torch.tensor([[0]], dtype=torch.float).to(cuda)

    # problems and lemmas are one-hot encoded. problems are encoded using the
    # first 100 indices, while lemmas are encoded using the last 100.
    # line_tensor is a concatenation of the problem w the lemma.

    prob_tensor = lineToTensor(pl[2], all_letters, cuda)
    zero_tensor = torch.zeros(prob_tensor.size()).to(cuda)
    prob_tensor = torch.cat((prob_tensor, zero_tensor), 2)
    
    lemma_tensor = lineToTensor(pl[3], all_letters, cuda)
    zero_tensor = torch.zeros(lemma_tensor.size()).to(cuda)
    lemma_tensor = torch.cat((zero_tensor, lemma_tensor), 2)
    
    line_tensor = torch.cat((prob_tensor, lemma_tensor), 0)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor
