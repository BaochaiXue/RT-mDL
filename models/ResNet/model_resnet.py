import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=10):
    model = models.resnet18(weights=None)  # Set weights to None instead of using pretrained
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet34(num_classes=10):
    model = models.resnet34(weights=None)  # Set weights to None instead of using pretrained
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
