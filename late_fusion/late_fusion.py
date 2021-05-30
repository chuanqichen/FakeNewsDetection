from transformers import BertForSequenceClassification, BertConfig
from transformers.models.bert import modeling_bert
import torch

bert_model = BertForSequenceClassification(BertConfig())
loaded = torch.load('../bert_model_save')
bert_model.load_state_dict(loaded)

