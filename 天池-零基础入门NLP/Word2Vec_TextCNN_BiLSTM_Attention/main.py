import random

import numpy as np
import torch
import logging
from train import *
from dataset import *
from utils import *
from config import *
from model import *
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def data_preprocess():
    rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    #用正则表达式按标点替换文本
    import re
    rawdata['words']=rawdata['text'].apply(lambda x: re.sub('3750|900|648',"",x))
    del rawdata['text']
    #数据划分
    #如果之前已经做了就直接加载
    if os.path.exists(test_index_file) and os.path.exists(train_index_file):
        test_index=joblib.load(test_index_file)
        train_index=joblib.load(train_index_file)
    else:
        rawdata.reset_index(inplace=True, drop=True)
        X = list(rawdata.index)
        y = rawdata['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                            stratify=y)  # stratify=y表示分层抽样，根据不同类别的样本占比进行抽样
        test_index = {'X_test': X_test, 'y_test': y_test}
        joblib.dump(test_index, 'test_index.pkl')
        train_index = {'X_train': X_train, 'y_train': y_train}
        joblib.dump(train_index, 'train_index.pkl')

    train_x=rawdata.loc[train_index['X_train']]['words']
    train_y=rawdata.loc[train_index['X_train']]['label'].values

    #sfolder=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
    # train_texts = []
    # train_labels = []
    # i=0
    # for train, label in sfolder.split(train_x,train_y):
    #     train_texts.extend(train.tolist())
    #     train_labels.extend(label)
    #     i+=1
    #     if i==9: #开发集
    #         dev_data = {'label': label, 'text': train}
    #         break


    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1,
                                                        stratify=train_y)
    train_data = {'label': y_train, 'text': X_train.values}
    dev_data = {'label': y_test, 'text': X_test.values}
    #测试集
    test_x=rawdata.loc[test_index['X_test']]
    test_y=rawdata.loc[test_index['X_test']]['label'].values
    test_data={'label': test_y, 'text': test_x['words'].tolist()}
    #预测

    f = pd.read_csv(final_test_data_file, sep='\t', encoding='UTF-8')
    final_test_data = f['text'].apply(lambda x: re.sub('3750|900|648',"",x))
    final_test_data = {'label': [0] * len(final_test_data), 'text': final_test_data.values}

    return train_data,dev_data,test_data,final_test_data

if os.path.exists('train_data.pkl'):
    train_data=joblib.load('train_data.pkl')
    dev_data = joblib.load('dev_data.pkl')
    test_data = joblib.load('test_data.pkl')
    final_test_data = joblib.load('final_test_data.pkl')
else:
    train_data, dev_data, test_data, final_test_data = data_preprocess()
    joblib.dump(train_data, 'train_data.pkl')
    joblib.dump(dev_data, 'dev_data.pkl')
    joblib.dump(test_data, 'test_data.pkl')
    joblib.dump(final_test_data, 'final_test_data.pkl')


log=Logger(mode='w')
log.logger.info("Dataset has built.")
vocab=Vocab(train_data)
log.logger.info("Vocab has built.")

log.logger.info("Creating Model.")
model=Model(log,vocab)
log.logger.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)
# train
# trainer = Trainer(log,model, vocab,train_data,dev_data,test_data,final_test_data)
# trainer.train()

# test
trainer = Trainer(log,model, vocab,final_test_data=final_test_data)
# trainer.test(flag=2)
trainer.test(flag=3)
