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
