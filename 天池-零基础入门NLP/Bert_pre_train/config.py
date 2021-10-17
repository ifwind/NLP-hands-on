dropout = 0.15
data_file = '../train_set.csv'

test_index_file='../test_index.pkl'
train_index_file='../train_index.pkl'
final_test_data_file = '../test_a.csv'


import torch
gpu=0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")

