# Task4-基于深度学习的文本分类3-基于Bert预训练和微调进行文本分类

因为天池这个比赛的数据集是脱敏的，无法利用其它已经预训练好的模型，所以需要针对这个数据集自己从头预训练一个模型。

我们利用Huggingface的transformer包，按照自己的需求从头开始预训练一个模型，然后将该模型应用于下游任务。

完整代码见：[NLP-hands-on/天池-零基础入门NLP at main · ifwind/NLP-hands-on (github.com)](https://github.com/ifwind/NLP-hands-on/tree/main/天池-零基础入门NLP/Bert_pre_train)

**注意：利用Huggingface做预训练需要安装wandb包，如果报错可参考**：[wandb.errors.UsageError: api_key not configured (no-tty). call wandb.login(key=\[your_api_key\\])_](https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/116124285)

## 预训练模型

利用Huggingface的transformer包进行预训练主要包括以下几个步骤：

1. 用数据集训练Tokenizer；
2. 加载数据及数据预处理；
3. 设定预训练模型参数，初始化预训练模型；
4. 设定训练参数，加载训练器；
5. 训练并保存模型。

### 用数据集训练Tokenizer

Tokenizer是分词器，分词方式有很多种，可以按照空格直接切分、也可以在按词组划分等，可以查看HuggingFace关于[tokenizers](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.models)的官方文档。

Huggingface中，Tokenizer的训练方式为：

1. 根据`tokenizers.models`实例化一个`Tokenizer`对象`tokenizer`，
2. 从`tokenizers.trainers`中选模型相应的训练器实例化，得到`trainer`；
3. 从`tokenizers.pre_tokenizers` 选定一个预训练分词器，对`tokenizer`的预训练分词器实例化；
4. 利用`tokenizer.train()`结合`trainer`对语料（**注意，语料为一行一句**）进行训练；
5. 利用`tokenizer.save()`保存`tokenizer`。

因为天池这个比赛的数据集是脱敏的，词都是用数字进行表示，没有办法训练wordpiece等复杂形式的分词器，只能用空格分隔，在wordlevel进行分词。

因此，我们利用`tokenizers.models`中的`WordLevel`模型，对应`tokenizers.trainers`中的`WordLevelTrainer`，选择预训练分词器为`Whitespace`训练分词器。

另外，在训练Tokenizer时，可以利用上全部的语料（包括训练集和最终的测试集）。

完整代码如下：

```python
import joblib
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
def data_preprocess():
    rawdata = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    #用正则表达式按标点替换文本
    import re
    rawdata['words']=rawdata['text'].apply(lambda x: re.sub('3750|900|648',"",x))
    del rawdata['text']

    #预测
    final_test_data = pd.read_csv(final_test_data_file, sep='\t', encoding='UTF-8')
    final_test_data['words'] = final_test_data['text'].apply(lambda x: re.sub('3750|900|648',"",x))
	del final_test_data['text']
    all_value= rawdata['words'].append(final_test_data['words'])
    all_value.columns=['text']
    all_value.to_csv('../alldata.csv',index=False)

data_preprocess()

from tokenizers import Tokenizer
from tokenizers.models import BPE,WordLevel
tokenizer= Tokenizer(WordLevel(unk_token="[UNK]"))

from tokenizers.trainers import BpeTrainer,WordLevelTrainer
#加入一些特殊字符
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

#空格分词器
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

#保存语料库文件
tokenizer.train(['../alldata.csv'], trainer)
tokenizer.mask_token='[MASK]'
tokenizer.save("../tokenizer-my-Whitespace.json")
```

### 加载数据及数据预处理

在预训练模型时，利用的是普通的、不带标签的句子，因此，在预训练模型时，同样采用全部的语料（包括训练集和最终的测试集，也就是前面的`alldata.csv`）。

使用Huggingface的datasets包的load_dataset函数加载数据，这里用的是csv的数据格式，关于其他格式的数据可以参考之前的一篇博客：[加载数据](https://ifwind.github.io/2021/08/26/BERT实战——（1）文本分类/#加载数据)。

输入模型之前，还需要对句子进行分词，转换成word id，此外，还需要padding长度不足的句子、获取对padding部分掩码的矩阵等等操作，这些由tokenizer进行处理，如果还需要其他预处理操作，都可以通过定义一个函数，将预处理操作封装起来，然后利用dataset.map()进行处理。

另外我们还在这里实例化了一个**数据收集器data_collator**，将经预处理的输入分batch再次处理后喂给模型。这是**为了告诉`Trainer`如何从预处理的输入数据中构造batch。**

```python
from transformers import PreTrainedTokenizerFast
#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'
#加载数据
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train':'alldata.csv'},cache_dir='Bert_pre_train\\') #这里的cache_dir考虑用绝对路径，不然可能会卡住
#预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True,max_length=512)
encoded_dataset = dataset.map(preprocess_function, batched=True)
#数据收集器
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15 #mlm表示是否使用masked language model；mlm_probability表示mask的几率
)
```

### 设定预训练模型参数，初始化预训练模型

这里选择了XLNet模型作为预训练模型的基础架构，然后根据调整模型的参数，也可以选 Roberta、ALBert等模型，参考[官方文档](https://huggingface.co/transformers/model_doc/xlnet.html)进行配置。

```python
from transformers import RobertaConfig,AlbertConfig,XLNetConfig
# 将模型的配置参数载入
config_kwargs = {
    "d_model": 512,
    "n_head": 4,
    "vocab_size": tokenizer.get_vocab_size(), # 自己设置词汇大小
    "embedding_size":64,
    "bi_data":True,
    "n_layer":8
}
config = XLNetConfig(**config_kwargs)
# 载入预训练模型，这里其实是根据某个模型结构调整config然后创建模型
from transformers import RobertaForMaskedLM,AlbertForMaskedLM,XLNetLMHeadModel

model = XLNetLMHeadModel(config=config)
```

### 设定训练参数，加载训练器

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./BERT",
    overwrite_output_dir=True,
    num_train_epochs=1, #训练epoch次数
    per_gpu_train_batch_size=12, #训练时的batchsize
    save_steps=10_000, #每10000步保存一次模型
    save_total_limit=2,#最多保存两次模型
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, #数据收集器在这里
    train_dataset=encoded_dataset["train"] #注意这里选择的是预处理后的数据集
)
```

### 训练并保存模型

```python
#开始训练
trainer.train()
#保存模型
trainer.save_model("./BERT")
```

## 微调模型进行分类

主要包括以下几个步骤：

1. 训练集划分；
2. 数据预处理；
3. 加载预训练模型、设置微调参数；
4. 微调训练下游任务模型并保存。

### 训练集划分

去掉可能的标点符号，并把当前竞赛给的**训练集划分为三个部分：训练集、验证集、测试集**。其中，训练集用于训练，验证集用于调参，测试集用于评估线下和线上的模型效果。

这里首先用train_test_split（注意使用分层抽样）把训练集划分为训练集和测试集（9：1），然后再将训练集进一步划分为训练集和开发集（9：1）。

```python
import joblib
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
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

    train_x=rawdata.loc[train_index['X_train']]
    train_y=rawdata.loc[train_index['X_train']]['label'].values

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1,
                                                        stratify=train_y)
    #训练集
    X_train.columns=['label', 'text']
    X_train.to_csv('train_data.csv',index=False)
    #开发集
    X_test.columns=['label', 'text']
    X_test.to_csv('dev_data.csv',index=False)
    #测试集
    test_x=rawdata.loc[test_index['X_test']]
    test_x.columns=['label', 'text']
    test_x.to_csv('test_data.csv',index=False)
```

### 数据预处理

和预训练模型时一致，加载数据集后还需要对数据进行预处理，注意上一步中加入了`label`这个字段，作为句子的分类标签。

**Huggingface封装了模型的有效字段，如果字段名称对不上会在`trainer`的`self.get_train_dataloader()`去掉无效字段，有效字段可以在debug模式下，在`transformer`的`trainer.py`的`self.args.remove_unused_columns`中查看。**

```python
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train':'train_data.csv',
                                          'dev':'dev_data.csv', 'test':'test_data.csv'},cache_dir='fine-tune\\') #这里的cache_dir考虑用绝对路径，不然可能会卡住
from transformers import PreTrainedTokenizerFast
#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True,max_length=512)
encoded_dataset = dataset.map(preprocess_function, batched=True)
```

### 加载预训练模型

```python
model_checkpoint = "BERT" #所选择的预训练模型

num_labels = 14
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```

### 设置微调参数

```python
metric_name = "acc"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name, #根据哪个评价指标选最优模型
    save_steps=10_000,
    save_total_limit=2,
)
from datasets import load_metric
import numpy as np
def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels,average='macro')
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


```

### 微调训练下游任务模型并保存

```python
trainer.train()
trainer.save_model("./test-glue")
```

### 评估模型

这里先用之前训练过程中保存的`test-glue/checkpoint-13500`模型进行评估，看一下训练效果，F1可以达到91.39左右。

```python
model_checkpoint = "test-glue/checkpoint-13500"#"BERT" #所选择的预训练模型

num_labels = 14
from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
batch_size = 12
metric_name = "acc"

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name, #根据哪个评价指标选最优模型
    save_steps=10_000,
    save_total_limit=2,
)
from datasets import load_metric
import numpy as np
def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels,average='macro')
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()
```

## 参考资料

[ALBERT — transformers 4.11.3 documentation (huggingface.co)](https://huggingface.co/transformers/model_doc/albert.html)

[BERT相关——（5）Pre-train Model | 冬于的博客 (ifwind.github.io)](https://ifwind.github.io/2021/08/22/BERT相关——（5）Pre-train Model/#重新配置模型)

[BERT实战——（1）文本分类 | 冬于的博客 (ifwind.github.io)](https://ifwind.github.io/2021/08/26/BERT实战——（1）文本分类/#定义评估方法)

[阅读源码-理解pytorch_pretrained_bert中BertTokenizer工作方式_枪枪枪的博客-CSDN博客](https://blog.csdn.net/az9996/article/details/109219652)

[NLP学习1 - 使用Huggingface Transformers框架从头训练语言模型 - 简书 (jianshu.com)](https://www.jianshu.com/p/fc3b80a64fa8)