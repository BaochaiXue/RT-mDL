import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet34(num_classes=10):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
