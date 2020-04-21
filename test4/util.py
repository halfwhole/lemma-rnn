import pdb
import random
import torch

# ENCODING SCHEME
# - is_var
#   - var_encoding: 1-49
# - is_dist
#   - dist_encoding: 0-64
# - is_const
#   - const_encoding: 1-335
# - is_func
#   - func_encoding: 1-5010
# - is_eq
#   - eq_encoding: 0/1
# - is_disj
#   - disj_name_encoding: 1-3332
#   - disj_role_encoding: 1-3
# - is_conj
# - is_(
# - is_)
# - is_special_symbol (?)

tensor_length = 16

def tokenToVec(token, cuda):
    tensor = torch.zeros(1, tensor_length)
    if token == '(':
        tensor[0][14] = 1; return tensor.to(cuda)
    elif token == ')':
        tensor[0][15] = 1; return tensor.to(cuda)
    elif token == 'Conj':
        tensor[0][13] = 1; return tensor.to(cuda)

    split = token.split(' ')
    keyword = split[0]

    if keyword == 'Var':
        tensor[0][0] = 1
        tensor[0][1] = int(split[1])
    elif keyword == 'Dist':
        tensor[0][2] = 1
        tensor[0][3] = int(split[1])
    elif keyword == 'Const':
        tensor[0][4] = 1
        tensor[0][5] = int(split[1])
    elif keyword == 'Func':
        tensor[0][6] = 1
        tensor[0][7] = int(split[1])
    elif keyword == 'Eq':
        tensor[0][8] = 1
        tensor[0][9] = int(split[1])
    elif keyword == 'Disj':
        tensor[0][10] = 1
        tensor[0][11] = int(split[1])
        tensor[0][12] = int(split[2])

    return tensor.to(cuda)

# Turn a line of tokens into a <line_length x 1 x N>
def lineToTensor(line, cuda):
    listTensors = [tokenToVec(token, cuda) for token in line]
    return torch.stack(listTensors).to(cuda)

def getTrainingExample(pl, usefulness, cuda):
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

    prob_tensor = lineToTensor(pl[2], cuda)
    zero_tensor = torch.zeros(prob_tensor.size()).to(cuda)
    prob_tensor = torch.cat((prob_tensor, zero_tensor), 2)

    lemma_tensor = lineToTensor(pl[3], cuda)
    zero_tensor = torch.zeros(lemma_tensor.size()).to(cuda)
    lemma_tensor = torch.cat((zero_tensor, lemma_tensor), 2)
    
    line_tensor = torch.cat((prob_tensor, lemma_tensor), 0)
    return pl_probname, pl_lemmaname, usefulness_tensor, line_tensor

if __name__ == '__main__':
    from parse import get_data
    from cuda_check import device
    usefulness, problemlemmas_test, problemlemmas_validation = get_data()
    pl = problemlemmas_test[0]
    x = getTrainingExample(pl, usefulness, device)
