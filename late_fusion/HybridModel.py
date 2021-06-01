import torch
from torch import nn
from torch.nn import Module
from transformers import BertModel, BertForSequenceClassification
from torchvision.models import ResNet


class LateFusionModel(Module):
    """Late fusion model for text + image

    Args:
        resnet_model (ResNet): a pretrained resnet50 instance
        bert_model (BertModel): a pretrained Bert instance
    """

    def __init__(self, resnet_model, bert_model):
        super(LateFusionModel, self).__init__()
        #assert isinstance(bert_model, BertModel), "Bert model must be a BertModel (e.g. sequence_classifier.bert)"
        assert isinstance(bert_model, BertForSequenceClassification)
        assert isinstance(resnet_model, ResNet), "resnet model must be a ResNet instance!"
        # --- modify the resnet model
        resnet_feature_size = resnet_model.fc.in_features
        self._resnet = resnet_model
        self._resnet.fc = nn.Identity()
        # Freeze resnet
        for param in self._resnet.parameters():
            param.requires_grad = False
        # ---- Set up the bert model for inference ---
        # bert_model.config.output_hidden_states = True
        self._bert = bert_model.bert
        bert_feature_size = bert_model.classifier.in_features
        #self._bert.classifier = nn.Identity()
        self._bert.eval()
        # Freeze bert
        for param in self._bert.parameters():
            param.requires_grad = False
        # Create the last linear layer
        self.linear = nn.Linear(bert_feature_size + resnet_feature_size, 1)

    def forward(self, batch_in:dict):
        """ Forward process

        :param input: input data as a dictionary with keys
        """
        #print(input)
        batch_in = {x: batch_in[x].to(next(self.parameters()).device) for x in batch_in}
        bert_output = self._bert(batch_in['bert_input_id'].squeeze(), attention_mask=batch_in['bert_attention_mask'].squeeze())
        cls_vector = bert_output.pooler_output
        resnet_feature = self._resnet(batch_in['image'])
        #print(cls_vector.size(), resnet_output.size())
        return self.linear(torch.cat((cls_vector, resnet_feature), dim=1))
