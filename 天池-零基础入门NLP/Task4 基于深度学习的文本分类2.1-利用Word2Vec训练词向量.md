# Task4 基于深度学习的文本分类2-Word2Vec

## 文本表示方法 Part2-2

### Word2Vec原理

> Word2Vec是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括CBOW和Skip-gram模型。 CBOW的方式是在知道词![[公式]](https://www.zhihu.com/equation?tex=w_t)的上下文![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt-2%7D%2Cw_%7Bt-1%7D%2Cw_%7Bt%2B1%7D%2Cw_%7Bt%2B2%7D)的情况下预测当前词![[公式]](https://www.zhihu.com/equation?tex=w_t).而Skip-gram是在知道了词![[公式]](https://www.zhihu.com/equation?tex=w_t)的情况下,对词![[公式]](https://www.zhihu.com/equation?tex=w_t)的上下 文![[公式]](https://www.zhihu.com/equation?tex=w_%7Bt-2%7D%2C+w_%7Bt-1%7D%2C+w_%7Bt%2B1%7D%2C+w_%7Bt%2B2%7D)进行预测，如下图所示：
>
> ![img](https://pic3.zhimg.com/80/v2-7339e1444995c19f962c900cf8c67106_720w.jpg)

关于word2vec的原理，觉得这个回答非常完整和清晰，分享一下~[深入浅出Word2Vec原理解析 - Microstrong的文章 - 知乎](https://zhuanlan.zhihu.com/p/114538417) 

## 训练基于Word2Vec的word embedding

### 数据加载、预处理和划分

这里的操作和之前一致，不再赘述。

```python
#数据加载、预处理
import pandas as pd
import joblib
data_file = 'train_set.csv'
rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
#用正则表达式按标点替换文本
import re
rawdata['words']=rawdata['text'].apply(lambda x: re.sub('3750|900|648',"",x))
del rawdata['text']

#数据划分
#如果之前已经做了就直接加载
test_data=joblib.load('test_index.pkl')
train_data=joblib.load('train_index.pkl')

#数据划分
from sklearn.model_selection import train_test_split
import joblib
rawdata.reset_index(inplace=True,drop=True)
X=list(rawdata.index)
y=rawdata['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,stratify=y) #stratify=y表示分层抽样，根据不同类别的样本占比进行抽样
test_data={'X_test':X_test,'y_test':y_test}
joblib.dump(test_data,'test_index.pkl')
train_data={'X_train':X_train,'y_train':y_train}
joblib.dump(train_data,'train_index.pkl')

train_x=rawdata.loc[train_data['X_train']]['words']
train_y=rawdata.loc[train_data['X_train']]['label'].values
test_x=rawdata.loc[test_data['X_test']]['words']
test_y=rawdata.loc[test_data['X_test']]['label'].values
```

### 使用gensim训练word2vec

本DEMO只使用部分数据，使用全部数据预训练的词向量地址：  

链接: https://pan.baidu.com/s/1ewlck3zwXVQuAzraZ26Euw 提取码: qbpr 


```python
import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed 
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
```

```python
logging.info('Start training...')
from gensim.models.word2vec import Word2Vec

num_features = 100     # Word vector dimensionality
num_workers = 4       # Number of threads to run in parallel

train_texts = list(map(lambda x: list(x.split()), train_x))
model = Word2Vec(train_texts, workers=num_workers, vector_size=num_features)
model.init_sims(replace=True)

# save model
model.save("./word2vec.bin")
```

```python
# load model
model = Word2Vec.load("./word2vec.bin")

# convert format
model.wv.save_word2vec_format('./word2vec.txt', binary=False)
```

## 参考资料

[Datawhale零基础入门NLP赛事 - Task5 基于深度学习的文本分类2-1Word2Vec](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.9.6406111apQ2nRk&postId=118268)

[深入浅出Word2Vec原理解析 - Microstrong的文章 - 知乎](https://zhuanlan.zhihu.com/p/114538417) 
