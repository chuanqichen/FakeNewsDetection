import torch

# Only works with transformers 3.5.1
loaded_bert = torch.load('../bert_model_save', map_location=torch.device('cpu'))
loaded_bert.save_pretrained('../bert_save_dir/')

