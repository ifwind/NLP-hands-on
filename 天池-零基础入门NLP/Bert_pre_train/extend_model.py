from transformers import XLNetPreTrainedModel, XLNetModel
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_utils import SequenceSummary


@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Model(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = XLNetModel(config)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,# 输出的维度
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )
        self.attn = nn.MultiheadAttention(embed_dim=config.d_model,num_heads=1,dropout=config.dropout)
        # self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]
        output, _ = self.lstm(output)
        attn_output, attn_output_weights = self.attn(output, output, output)
        # output = self.sequence_summary(attn_output)
        logits = self.logits_proj(attn_output[:,0,:])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            # attentions=attn_output_weights,
        )
if __name__ == '__main__':

    from datasets import load_metric
    import numpy as np
    def compute_metrics(eval_pred):
        metric = load_metric('f1')
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels,average='macro')

    from datasets import load_dataset
    dataset = load_dataset('csv', data_files={'train':'train_data.csv',
                                              'dev':'dev_data.csv',
                                              'test':'test_data.csv'},cache_dir='D:\\NLP\\Bert_pre_train\\fine-tune\\')
    from transformers import PreTrainedTokenizerFast
    #注意这里用了另外一种方式加载Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-my-Whitespace.json")
    tokenizer.mask_token='[MASK]'
    tokenizer.pad_token='[PAD]'

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True,max_length=512)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    from transformers import Trainer, TrainingArguments

    model_checkpoint = "./pre-train2" #所选择的预训练模型
    num_labels = 14
    model=Model.from_pretrained(model_checkpoint, num_labels=num_labels)

    batch_size = 12
    metric_name = "f1"

    args = TrainingArguments(
        "extend-XLNet",
        evaluation_strategy = "epoch", #每个epcoh会做一次验证评估；
        # eval_steps =10,
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
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    #
    # trainer.train()
    trainer.save_model("./extend-XLNet")
    trainer.evaluate(encoded_dataset["test"])
    # import pandas as pd
    # final_test_dataset = load_dataset('csv', data_files={'train':'final_test_data.csv'},cache_dir='D:\\NLP\\Bert_pre_train\\fine-tune\\')
    # encoded_final_test_dataset=final_test_dataset.map(preprocess_function, batched=True)
    # res=trainer.predict(test_dataset=encoded_final_test_dataset["train"])
    # csv=pd.DataFrame(np.argmax(res[0],1),columns=['label'])
    # csv.to_csv('res.csv' ,index=False)
    # print('end')