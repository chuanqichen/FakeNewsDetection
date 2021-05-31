import torch
from transformers import BertForSequenceClassification
from resnet.my_resnet import resnet50_2way

# Load bert model
bert_model = BertForSequenceClassification.from_pretrained('../bert_save_dir')
print(bert_model)

# Load resnet
resnet_model = resnet50_2way(pretrained=False)
resnet_dict = torch.load('../fakeddit_resnet.pt', map_location=torch.device('cpu'))
resnet_model.load_state_dict(resnet_dict)
print(resnet_model)