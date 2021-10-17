#加载训练数据
import joblib
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
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
    #预测
    f = pd.read_csv(final_test_data_file, sep='\t', encoding='UTF-8')
    final_test_data = f['text'].apply(lambda x: re.sub('3750|900|648',"",x))
    final_test_data.to_csv('final_test_data.csv',index=False)
from datasets import load_metric
import numpy as np
def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels,average='macro')
if __name__ == '__main__':

    if not os.path.exists('final_test_data.csv'):
        data_preprocess()

    from datasets import load_dataset
    dataset = load_dataset('csv', data_files={'train':'train_data.csv',
                                              'dev':'dev_data.csv',
                                              'test':'test_data.csv'},cache_dir='D:\\NLP\\Bert_pre_train\\fine-tune\\') #这里建议用完全路径，否则可能卡住

    from transformers import PreTrainedTokenizerFast
    #注意这里用了另外一种方式加载Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-my-Whitespace.json")
    tokenizer.mask_token='[MASK]'
    tokenizer.pad_token='[PAD]'

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True,max_length=512)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    from transformers import Trainer, TrainingArguments

    model_checkpoint = "BERT"#"BERT" #所选择的预训练模型

    num_labels = 14
    from transformers import XLNetForSequenceClassification, TrainingArguments, Trainer
    model = XLNetForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    batch_size = 12
    metric_name = "f1"

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

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./test-glue")
    # trainer.evaluate()