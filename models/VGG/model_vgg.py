import torch.nn as nn
import torchvision.models as models

def get_vgg11(num_classes=10):
    model = models.vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_vgg13(num_classes=10):
    model = models.vgg13(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_vgg16(num_classes=10):
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_vgg19(num_classes=10):
    model = models.vgg19(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
