import torch
from transformers import BertForSequenceClassification

bert_model = BertForSequenceClassification.from_pretrained('../bert_save_dir')
print(bert_model)
