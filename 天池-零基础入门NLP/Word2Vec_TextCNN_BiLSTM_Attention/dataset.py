import random

import numpy as np
import torch
import logging
# build vocab
from collections import Counter
import os
from config import *
import joblib
# Vocab 的作用是：
# 1. 创建 词 和 index 对应的字典，这里包括 2 份字典，分别是：_id2word 和 _id2extword
# 其中 _id2word 是从新闻得到的， 把词频小于 5 的词替换为了 UNK。对应到模型输入的 batch_inputs1。
# _id2extword 是从 word2vec.txt 中得到的，有 5976 个词。对应到模型输入的 batch_inputs2。
# 后面会有两个 embedding 层，其中 _id2word 对应的 embedding 是可学习的，_id2extword 对应的 embedding 是从文件中加载的，是固定的
# 2.创建 label 和 index 对应的字典

class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        #创建词和 index 对应的字典
        self._word2id = reverse(self._id2word)
        #创建 label 和 index 对应的字典
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    #创建词典
    def build_vocab(self, data):
        if os.path.exists(save_word_counter_path):
            self.word_counter=joblib.load(save_word_counter_path)
        else:
            self.word_counter = Counter()
            #计算每个词出现的次数
            for text in data['text']:
                words = text.split()
                self.word_counter+=Counter(words)
            joblib.dump(self.word_counter,save_word_counter_path)
            # for word in words:
            #     self.word_counter[word] += 1
        # 去掉频次小于 min_count = 5 的词，把词存到 _id2word
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label] # 取出 label 对应的次数
            self._id2label.append(label)
            self.target_names.append(label2name[label]) # 根据label数字取出对应的名字

    def load_pretrained_embs(self, embfile,save_embfile):
        if os.path.exists(save_embfile):
            embeddings= joblib.load(save_embfile)
            self._id2extword=embeddings['id2extword']
            embeddings= embeddings['embeddings']
        else:
            with open(embfile, encoding='utf-8') as f:
                lines = f.readlines()
                items = lines[0].split()
                # 第一行分别是单词数量、词向量维度
                word_count, embedding_dim = int(items[0]), int(items[1])

            index = len(self._id2extword)
            embeddings = np.zeros((word_count + index, embedding_dim))
            # 下面的代码和 word2vec.txt 的结构有关
            for line in lines[1:]:
                values = line.split()
                self._id2extword.append(values[0]) # 首先添加第一列的单词
                vector = np.array(values[1:], dtype='float64') # 然后添加后面 100 列的词向量
                embeddings[self.unk] += vector
                embeddings[index] = vector
                index += 1

            # unk 的词向量是所有词的平均
            embeddings[self.unk] = embeddings[self.unk] / word_count
            # 除以标准差干嘛？
            embeddings = embeddings / np.std(embeddings)
            joblib.dump({"embeddings":embeddings,"id2extword":self._id2extword}, save_embfile)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    # 根据单词得到 id
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)
    # 根据单词得到 ext id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)
    # 根据 label 得到 id
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)
# 作用是：根据一篇文章，把这篇文章分割成多个句子
# text 是一个新闻的文章
# vocab 是词典
# max_sent_len 表示每句话的长度
# max_segment 表示最多有几句话
# 最后返回的 segments 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):

    words = text.strip().split()
    document_len = len(words)
    # 划分句子的索引，句子长度为 max_sent_len
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        # 根据索引划分句子
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        # 把出现太少的词替换为 UNK
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        # 添加 tuple:(句子长度，句子本身)
        segments.append([len(segment), segment])

    assert len(segments) > 0
    # 如果大于 max_segment 句话，则句数减少一半，返回一半的句子
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        # 否则返回全部句子
        return segments

# 最后返回的数据是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # sents_words: 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            # 把 word 转为 id
            word_ids = vocab.word2id(sent_words)
            # 把 word 转为 ext id
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])

    return examples
# data 参数就是 get_examples() 得到的
# data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        # 这里是打乱所有数据
        np.random.shuffle(data)
        # lengths 表示的是 每篇文章的句子数量
        lengths = [example[1] for example in data]
        noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
        sorted_indices = np.argsort(noisy_lengths).tolist()
        sorted_data = [data[i] for i in sorted_indices]
    else:
        sorted_data = data
    # 把 batch 的数据放进一个 list
    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        # 打乱 多个 batch
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch
# build loader
# data 参数就是 get_examples() 得到的
# data是一个 list，每个元素是一个 tuple: (label, 句子数量，doc)
# 其中 doc 又是一个 list，每个 元素是一个 tuple: (句子长度，word_ids, extword_ids)
def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        # 如果 i < batch_num - 1，那么大小为 batch_size，否则就是最后一批数据
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs

if __name__ == '__main__':
    import pandas as pd
    import joblib

    data_file = '../train_set.csv'
    rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    # 用正则表达式按标点替换文本
    import re

    rawdata['words'] = rawdata['text'].apply(lambda x: re.sub('3750|900|648', "", x))
    del rawdata['text']
    # 数据划分
    # 如果之前已经做了就直接加载
    test_index = joblib.load('../test_index.pkl')
    train_index = joblib.load('../train_index.pkl')
    train_x = rawdata.loc[train_index['X_train']]['words']
    train_y = rawdata.loc[train_index['X_train']]['label'].values

    from sklearn.model_selection import StratifiedKFold

    sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    train_texts = []
    train_labels = []

    i = 0
    for train, label in sfolder.split(train_x, train_y):
        train_texts.extend(train.tolist())
        train_labels.extend(label)
        i += 1
        if i == 9:  # 开发集
            dev_data = {'label': label, 'text': train}
            break
    train_data = {'label': train_labels, 'text': train_texts}

    # 测试集
    test_x = rawdata.loc[test_index['X_test']]
    test_y = rawdata.loc[test_index['X_test']]['label'].values
    test_data = {'label': test_y, 'text': test_x['words'].tolist()}
    # 预测
    final_test_data_file = '../test_a.csv'
    f = pd.read_csv(final_test_data_file, sep='\t', encoding='UTF-8')
    final_test_data = f['text'].apply(lambda x: re.sub('3750|900|648', "", x))
    final_test_data = {'label': [0] * len(final_test_data), 'text': final_test_data.tolisit()}

    vocab = Vocab(train_data)