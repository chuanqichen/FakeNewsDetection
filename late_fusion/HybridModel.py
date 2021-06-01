from torch import nn
from torch.nn import Module
from transformers import BertModel
from torchvision.models import ResNet


class LateFusionModel(Module):
    """Late fusion model for text + image

    Args:
        resnet_model (ResNet): a pretrained resnet50 instance
        bert_model (BertModel): a pretrained Bert instance
    """

    def __init__(self, resnet_model, bert_model):
        super(LateFusionModel, self).__init__()
        assert isinstance(bert_model, BertModel), "Bert model must be a BertModel (e.g. sequence_classifier.bert)"
        assert isinstance(resnet_model, ResNet), "resnet model must be a ResNet instance!"
        # --- modify the resnet model
        self._resnet = resnet_model
        self._resnet.fc = nn.Identity()
        # Freeze resnet
        for param in self._resnet.parameters():
            param.requires_grad = False
        # ---- Set up the bert model for inference ---
        # bert_model.config.output_hidden_states = True
        self._bert = bert_model
        self._bert.eval()
        # Freeze bert
        for param in self._bert.parameters():
            param.requires_grad = False
        # Create the last linear layer
        bert_feature_size = bert_model.pooler.out_features
        resnet_feature_size = resnet_model.fc.in_features
        self.linear = nn.Linear(bert_feature_size + resnet_feature_size, 1)

    def forward(self, input):
        """ Forward process

        :param input: input data as a dictionary with keys
        """
        bert_output = self._bert(input['bert_input_id'], attention_mask=input['bert_attention_mask'])
        cls_vector = bert_output[1]
        resnet_output = self._resnet(input['image'])
        print(cls_vector.size, resnet_output.size)
