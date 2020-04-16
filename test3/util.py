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
    # Concatenate the problem with lemma, seperated by unused '@'
    pl_probcatlemma = pl[2] + '@' + pl[3]
    pl_usefulness = usefulness[pl_probname][pl_lemmaname]
    if (pl_usefulness < 1):
        usefulness_tensor = torch.tensor([[1]], dtype=torch.float).to(cuda)
    else:
        usefulness_tensor = torch.tensor([[0]], dtype=torch.float).to(cuda)
    line_tensor = lineToTensor(pl_probcatlemma, all_letters, cuda)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor
