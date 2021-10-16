# 读取训练好的词向量文件
word2vec_path = '../word2vec.txt'
save_word2vec_embed_path='word2vec.emb'
save_word_counter_path='../word_counter.pkl'
dropout = 0.15

# build optimizer
learning_rate = 2e-4
decay = .75
decay_step = 1000
data_file = '../train_set.csv'

test_index_file='../test_index.pkl'
train_index_file='../train_index.pkl'
final_test_data_file = '../test_a.csv'

clip = 5.0
epochs = 10
early_stops = 3
log_interval = 50

test_batch_size = 256
train_batch_size = 256

save_model = './cnn.bin'
save_test = './cnn.csv'

import torch
gpu=0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")

