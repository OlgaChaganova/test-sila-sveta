import torch.nn as nn
from torchvision import models


def get_model(num_layers: int, layers_to_unfreeze: int = 4, num_classes: int = 3):
    assert num_layers in [121, 161, 169, 201], 'Number of layers must be one of the following: 121, 161, 169, 201'
    if num_layers == 121:
        densenet = models.densenet121(pretrained=True)
    elif num_layers == 161:
        densenet = models.densenet161(pretrained=True)
    if num_layers == 169:
        densenet = models.densenet169(pretrained=True)
    if num_layers == 201:
        densenet = models.densenet201(pretrained=True)

    # defining layers that won't be learnable ("freeze" them)
    for param in densenet.features[:-layers_to_unfreeze].parameters():
        param.requires_grad = False

    # change classifier for the needed number of classes
    densenet.classifier = nn.Linear(1024, num_classes)
    return densenet
