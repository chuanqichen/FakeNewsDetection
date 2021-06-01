from torchvision import models
from torch import nn


def resnet50_2way(pretrained: bool):
    """Return a customized resnet

    :param pretrained: whether pretrain or not
    :return: the model
    """
    model_ft = models.resnet50(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 1.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 1)
    return model_ft