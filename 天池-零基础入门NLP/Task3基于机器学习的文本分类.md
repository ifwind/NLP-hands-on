# **Task3 基于机器学习的文本分类**

实操主要包括以下几个任务：

1. 基于文本统计特征的特征提取（包括词频特征、TF-IDF特征等）
2. 如何划分训练集（用于参数选择、交叉验证）
3. 结合提取的不同特征和不同模型（线性模型、集成学习模型）完成训练和预测

## 文本表示方法

在机器学习算法的训练过程中，假设给定$N$个样本，每个样本有$M$个特征，这样组成了$N×M$的样本矩阵，然后完成算法的训练和预测。文本表示成计算机能够运算的数字或向量的方法一般称为词嵌入（Word Embedding）方法。词嵌入将不定长的文本转换到定长的空间内，是文本分类的第一步。

### One-hot

将每一个单词使用一个离散的向量表示。具体将每个字/词编码一个索引，然后根据索引进行赋值。

One-hot表示方法的例子如下：

```python
句子1：我 爱 北 京 天 安 门
句子2：我 喜 欢 上 海
```

首先对所有句子的字进行索引，即将每个字确定一个编号：

```python
{
	'我': 1, '爱': 2, '北': 3, '京': 4, '天': 5,
  '安': 6, '门': 7, '喜': 8, '欢': 9, '上': 10, '海': 11
}
```

在这里共包括11个字，因此每个字可以转换为一个11维度稀疏向量：

```
我：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
爱：[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
海：[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

### Bag of Words

Bag of Words（词袋表示），也称为Count Vectors，每个文档的字/词可以使用其出现次数（词频）来进行表示。

```python
句子1：我 爱 北 京 天 安 门
句子2：我 喜 欢 上 海
```

直接统计每个字出现的次数，并进行赋值：

```python
句子1：我 爱 北 京 天 安 门
转换为 [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

句子2：我 喜 欢 上 海
转换为 [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
```

在sklearn中可以直接`CountVectorizer`来实现这一步骤：

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus).toarray()
```

### N-gram

N-gram与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数。

如果N取值为2，则句子1和句子2就变为：

```
句子1：我爱 爱北 北京 京天 天安 安门
句子2：我喜 喜欢 欢上 上海
```

### TF-IDF

TF-IDF 分数由两部分组成：第一部分是**词语频率**（Term Frequency），第二部分是**逆文档频率**（Inverse Document Frequency）。其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率。

```
TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
IDF(t)= log_e（文档总数 / 出现该词语的文档总数）
```

## 机器学习模型

机器学习是对能通过经验自动改进的计算机算法的研究。机器学习通过历史数据**训练**出**模型**对应于人类对经验进行**归纳**的过程，机器学习利用**模型**对新数据进行**预测**对应于人类利用总结的**规律**对新问题进行**预测**的过程。

1. 机器学习能解决一定的问题，但不能奢求机器学习是万能的；
2. 机器学习算法有很多种，看具体问题需要什么，再来进行选择；
3. 每种机器学习算法有一定的偏好，需要具体问题具体分析；

本文主要为利用python进行实操，使用到的模型理论部分可以跳转相应的链接学习。

1. [线性模型](https://ifwind.github.io/2021/07/18/%E8%A5%BF%E7%93%9C%E4%B9%A6%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%E2%80%94%E2%80%94%E7%AC%AC3%E7%AB%A0-%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%EF%BC%883-1%E3%80%813-2%EF%BC%89/)
2. [Adaboost](https://ifwind.github.io/2021/09/27/%E8%A5%BF%E7%93%9C%E4%B9%A6%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%E2%80%94%E2%80%94%E7%AC%AC8%E7%AB%A0-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/#%E5%9F%BA%E4%BA%8E%E5%8A%A0%E6%80%A7%E6%A8%A1%E5%9E%8B%E7%9A%84adaboost)
3. [XGBoost](https://ifwind.github.io/2021/09/27/西瓜书阅读笔记——第8章-集成学习/#gbdt和xgboost)
4. [随机森林](https://ifwind.github.io/2021/09/27/%E8%A5%BF%E7%93%9C%E4%B9%A6%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0%E2%80%94%E2%80%94%E7%AC%AC8%E7%AB%A0-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97)
5. [LightGBM](https://zhuanlan.zhihu.com/p/406106014)

值得注意的是，XGBoost、LightGBM并不是指某个算法，而是机器学习算法GBDT的算法实现，高效地实现了GBDT算法并进行了算法和工程上的许多改进，具体区别和联系可参考：[GBDT、XGBoost、LightGBM的区别和联系](https://www.jianshu.com/p/765efe2b951a)

## 基于机器学习的文本分类实验

### 数据加载及预处理

根据数据分析结果，我们这里可以去掉可能是标点符号的字符（字符3750，字符900和字符648）。

```python
import pandas as pd
import joblib
data_file = 'train_set.csv'
rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
#用正则表达式按标点替换文本
import re
rawdata['words']=rawdata['text'].apply(lambda x: re.sub('3750|900|648',"",x))
del rawdata['text']
```

### 数据集划分

一般策略是：

1. **在选择模型、确定模型参数时**：

   先把当前竞赛给的**训练集划分为三个部分：训练集、验证集、测试集**。其中，训练集用于训练，验证集用于调参，测试集用于评估线下和线上的模型效果。

   特别的，为了进行交叉验证，划分后的训练集留10%用于测试；剩下90%输入交叉验证。

   注意：测试集最好固定下来，可以先打乱然后记录下索引值，之后直接取用。

   有一些模型的参数需要选择，这些参数会在一定程度上影响模型的精度，那么如何选择这些参数呢？

   - 通过阅读文档，要弄清楚这些参数的大致含义，那些参数会增加模型的复杂度
   - 通过在验证集上进行验证模型精度，找到模型在是否过拟合还是欠拟合

   ![Image](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/1095279501877/1594909879453_RrvunJz6cT.jpg)

2. **在确定最佳模型参数（best parameters）后**：可以把原始的完整训练集全部扔给设置为best parameters的模型进行训练，得到最终的模型（final model），然后利用final model对真正的测试集进行预测。

**注意：由于训练集中，不同类别的样本个数不同，在进行数据集划分时可以考虑使用分层抽样，根据不同类别的样本占比进行抽样，以保证标签的分布与整个数据集的分布一致。**

```python
from sklearn.model_selection import train_test_split
import joblib
rawdata.reset_index(inplace=True,drop=True)
test_data=joblib.load('test_index.pkl')
train_data=joblib.load('train_index.pkl')
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

利用StratifiedKFold对划分后的train_data分层抽样：

```python
sfolder=sklearn.model_selection.StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

for train, test in sfolder.split(X,y):
    print('Train: %s | test: %s' % (train, test))
```

```python
# test 读取测试集数据
test_data_file = 'test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
test_data = f['text'].apply(lambda x: re.sub('3750|900|648',"",x))
```

### 特征提取

使用sklearn库进行词频特征和TFIDF特征提取，注意，这里使用划分后的训练集提取各个词的特征(fit)，而由训练集划分出的测试集的特征是由该特征映射的，而不是用全部训练集提取的特征映射来的；而最终测试集的特征由全部训练集提取的特征映射得到。

max_features表示特征维度，该值也算是超参数，这里只设置了300维。

#### 词频特征

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=300)
vectorizer = vectorizer.fit(train_x)
train_text = vectorizer.transform(train_x)
#训练集划分出的测试集的特征由train_x映射
test_text = vectorizer.transform(test_x)
#最终测试集
vectorizer = CountVectorizer(max_features=300)
vectorizer = vectorizer.fit(rawdata['words'])
final_test_text=vectorizer.transform(test_data)
```

#### TF-IDF特征

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=300)
tfidf = tfidf.fit(train_x)
train_text_tfidf = tfidf.transform(train_x)
test_text_tfidf = tfidf.transform(test_x)
final_test_text_tfidf=tfidf.transform(test_data)
#最终测试集
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=300)
tfidf = tfidf.fit(rawdata['words'])
final_test_text=tfidf.transform(test_data)
```

### 词频特征+线性模型

通过本地构建验证集计算F1得分。


```python
# Count Vectors + RidgeClassifier

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

clf = RidgeClassifier()
clf.fit(train_text[:10000], train_y[:10000])

val_pred = clf.predict(train_text[10000:])
print(f1_score(train_y[10000:], val_pred, average='macro'))

```

    测试集精度：0.5948335699700965

### TFIDF 特征+线性模型

```python
# TF-IDF +  RidgeClassifier
import pandas as pd

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

clf = RidgeClassifier()
clf.fit(train_text_tfidf[:10000], train_y.values[:10000])

val_pred = clf.predict(train_text_tfidf[10000:])
print(f1_score(train_y.values[10000:], val_pred, average='macro'))
# 0.87
```

    0.6928858913248898



### TFIDF 特征+Adaboost

利用sklearn内置的Adaboost模块可以直接进行模型训练，Adaboost是集成学习模型，可以选择不同的个体学习器，这里选择决策树分类器作为基学习器。

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
bdt.fit(train_text_tfidf, train_y)

Z = bdt.predict(test_text_tfidf)
print "Score:", bdt.score(train_text_tfidf,train_y)

```



### TFIDF 特征+XGBoost

额外安装依赖库xgboost：

```python
pip install xgboost
#pip install xgboost==1.4.2
```

由两种方式可以利用XGBoost库进行模型训练和预测，第一种基于XGBoost的sklearn API（推荐），第二种是基于XGBoost自带接口。

```python
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

#基于XGB的sklearn API-推荐
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier(
    n_estimators=20,  # 迭代次数
    learning_rate=0.1,  # 步长
    max_depth=5,  # 树的最大深度
    min_child_weight=1,  # 决定最小叶子节点样本权重和
    silent=1,  # 输出运行信息
    subsample=0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
    colsample_bytree=0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
    objective='multi:softmax',  # 多分类！！！！！！
    num_class=3,
    nthread=4,
    seed=27)
print "training..."
clf.fit(train_text_tfidf, train_y, verbose=True)
fit_pred = clf.predict(test_text_tfidf)

# 基于XGBoost自带接口
# xgboost模型初始化设置
dtrain=xgb.DMatrix(train_text_tfidf,label=train_y)
dtest=xgb.DMatrix(test_text_tfidf)
watchlist = [(dtrain,'train')]

# booster:
params={'booster':'gbtree',
        'objective': 'multi:softmax',#多分类：'multi:softmax'，二分类：'binary:logistic',
        'num_class':14,#类别个数
        'eval_metric': 'auc',
        'max_depth':5,
        'lambda':10,
        'subsample':0.75,
        'colsample_bytree':0.75,
        'min_child_weight':2,
        'eta': 0.025,
        'seed':0,
        'nthread':8,
        'gamma':0.15,
        'learning_rate' : 0.01}

# 建模与预测：50棵树
bst=xgb.train(params,dtrain,num_boost_round=50,evals=watchlist)
ypred=bst.predict(dtest)
 
# 设置阈值、评价指标
y_pred = (ypred >= 0.5)*1
print ('Precesion: %.4f' %metrics.precision_score(test_y,ypred,average='macro'))
print ('Recall: %.4f' % metrics.recall_score(test_y,ypred,average='macro'))
print ('F1-score: %.4f' %metrics.f1_score(test_y,ypred,average='macro'))
print ('Accuracy: %.4f' % metrics.accuracy_score(test_y,ypred,average='macro'))
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred,average='macro'))
```



### TFIDF 特征+随机森林

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(train_text_tfidf,train_y)
print rf0.oob_score_
y_predprob = rf0.predict_proba(train_text_tfidf)[:,1]
```

### TFIDF+LightGBM

需要安装依赖库lightgbm。

```python
pip install lightgbm 
#pip install lightgbm==3.3.0
#LightGBM官方文档:http://lightgbm.readthedocs.io/en/latest/Python-Intro.html
```

lightgbm库和XGBoost库的使用方式类似，相比XGBoost的优点可参考[LightGBM核心解析与调参 - 掘金 (juejin.cn)](https://juejin.cn/post/6844903661160628231)。

这里给出推荐的使用方法：

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = LGBMClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=200, # 使用多少个弱分类器
    objective='multiclass',
    num_class=3,
    booster='gbtree',
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    seed=0 # 随机数种子
)
model.fit(train_text_tfidf,train_y, eval_set=[(train_text_tfidf, train_y), (test_text_tfidf, test_y)], 
          verbose=100, early_stopping_rounds=50)

# 对测试集进行预测
y_pred = model.predict(test_text_tfidf)

#计算准确率
accuracy = accuracy_score(test_y,y_pred)
print('accuracy:%3.f%%'%(accuracy*100))

# 显示重要特征
plot_importance(model)
plt.show()
#accuracy: 90%
```

### 交叉验证模型效果

```python
score = cross_val_score(clf,X,y,cv=skf,scoring='f1')

print('f1 score:'+str(score)+'\n')
```

### 交叉验证+网格搜索选择参数

结合分层抽样和网格搜索超参数来确定最优模型：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
 
decision_tree_classifier = DecisionTreeClassifier()
 
parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}
 
skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
skf = skf.get_n_splits(train_text_tfidf, train_y)
grid_search = GridSearchCV(decision_tree_classifier, param_grid=parameter_grid,cv=skf)#注意这里将cv设置为分层抽样交叉验证
grid_search.fit(all_inputs, all_classes)
```

这里以随机森林和LightGBM的交叉验证+网格搜索为例进行实验。

#### TFIDF+随机森林+交叉验证+网格搜索

```python
param_test1 = {'n_estimators':range(10,71,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), param_grid = param_test1,cv=skf)
gsearch1.fit(train_text_tfidf,train_y)
gsearch2.cv_results_,gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 70, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2,  cv=skf)#注意这里将cv设置为分层抽样交叉验证
gsearch2.fit(train_text_tfidf,train_y)
gsearch2.cv_results_,gsearch2.best_params_, gsearch2.best_score_
```

#### TFIDF+LightGBM+交叉验证+网格搜索

```python
from sklearn.datasets import load_iris
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split

parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = LGBMClassifier(max_depth=3,
                    learning_rate=0.1,
                    n_estimators=200, # 使用多少个弱分类器
                    objective='multiclass',
                    num_class=3,
                    booster='gbtree',
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0,
                    reg_lambda=1,
                    seed=0 # 随机数种子
                )
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=skf)
gsearch.fit(train_text_tfidf,train_y)

```



## 参考资料

[Datawhale零基础入门NLP赛事 - Task3 基于机器学习的文本分类](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.6406111apQ2nRk&postId=118254)

[GBDT、XGBoost、LightGBM的区别和联系](https://www.jianshu.com/p/765efe2b951a)

[集成学习（五）LightGBM](https://zhuanlan.zhihu.com/p/406106014)

[西瓜书阅读笔记——第8章-集成学习](https://ifwind.github.io/2021/09/27/西瓜书阅读笔记——第8章-集成学习/)

[西瓜书阅读笔记——第3章-线性回归（3.1-3.2）](https://ifwind.github.io/2021/07/18/西瓜书阅读笔记——第3章-线性回归（3-1、3-2）/)

[LightGBM 重要参数、方法、函数理解及调参思路、网格搜索（附例子）](https://blog.csdn.net/VariableX/article/details/107256149)

[LightGBM核心解析与调参](https://juejin.cn/post/6844903661160628231)

[Parameters — LightGBM 3.3.0.99 documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

