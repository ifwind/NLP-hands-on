from config import *
import pandas as pd
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
if __name__ == '__main__':

    if not os.path.exists('../alldata.csv'):
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