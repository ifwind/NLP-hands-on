
from transformers import PreTrainedTokenizerFast
#注意这里用了另外一种方式加载Tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-my-Whitespace.json")
tokenizer.mask_token='[MASK]'
tokenizer.pad_token='[PAD]'
#数据预处理
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train':'alldata.csv'},cache_dir='D:\\NLP\\Bert_pre_train\\')
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True,max_length=512)
encoded_dataset = dataset.map(preprocess_function, batched=True)
#数据收集器
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15 #mlm表示是否使用masked language model；mlm_probability表示mask的几率
)

#模型配置
# 自己修改部分配置参数
# config_kwargs = {
#     "cache_dir": None,
#     "revision": 'main',
#     "use_auth_token": None,
#     "hidden_size": 512,
#     "num_attention_heads": 2,
#     "hidden_dropout_prob": 0.2,
#     "vocab_size": tokenizer.get_vocab_size(), # 自己设置词汇大小
#     "embedding_size":64
# }
config_kwargs = {
    "d_model": 512,
    "n_head": 4,
    "vocab_size": tokenizer.vocab_size, # 自己设置词汇大小
    "embedding_size":64,
    "bi_data":True,
    "n_layer":8
}
# 将模型的配置参数载入
from transformers import RobertaConfig,AlbertConfig,XLNetConfig

config = XLNetConfig(**config_kwargs)
# 载入预训练模型，这里其实是根据某个模型结构调整config然后创建模型
from transformers import RobertaForMaskedLM,AlbertForMaskedLM,XLNetLMHeadModel

model = XLNetLMHeadModel(config=config)
# model = AutoModelForMaskedLM.from_pretrained(
#             'pretrained_bert_models/bert-base-chinese/',
#             from_tf=bool(".ckpt" in 'roberta-base'), # 支持tf的权重
#             config=config,
#             cache_dir=None,
#             revision='main',
#             use_auth_token=None,
#         )
# model.resize_token_embeddings(len(tokenizer))
#output:Embedding(863, 768, padding_idx=1)

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

#开始训练
trainer.train()
#保存模型
trainer.save_model("./BERT")