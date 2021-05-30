from transformers import BertForSequenceClassification, BertConfig
import torch

# bert_model = BertForSequenceClassification(BertConfig())
loaded_bert = torch.load('../bert_model_save', map_location=torch.device('cpu')) # this model was created in transformers 3.5.1
#bert_model.load_state_dict(loaded)
loaded_bert.save_pretrained('../bert_save_dir/')

