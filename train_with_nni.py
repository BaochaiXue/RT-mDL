import nni
import torch
import torch.nn as nn
import torch.optim as optim
from training_evaluation import train, test
from data_preparation import get_cifar10_loaders, get_gtsrb_loaders, get_self_collected_loaders
from models.AlexNet.model_alexnet import AlexNet
from models.ResNet.model_resnet import get_resnet18, get_resnet34
from models.VGG.model_vgg import get_vgg11, get_vgg13, get_vgg16, get_vgg19
from models.TinyYOLO.model_tiny_yolo import TinyYOLO
import torch.nn.utils.prune as prune
import time
import copy

def prune_model(model, amount):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

def weight_sharing(model):
    shared_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if module.weight.shape in shared_weights:
                module.weight = shared_weights[module.weight.shape]
            else:
                shared_weights[module.weight.shape] = module.weight
    return model

def width_scaling(model, factor):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_channels = int(module.out_channels * factor)
            module.out_channels = out_channels
    return model

def depth_scaling(model, factor):
    new_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            new_layers.append(module)
            if factor < 1.0:
                if len(new_layers) > 1:
                    new_layers.pop()
    model.features = nn.Sequential(*new_layers)
    return model

def main():
    # Get NNI parameters
    params = nni.get_next_parameter()
    learning_rate = params.get('learning_rate', 0.001)
    pruning_amount = params.get('pruning_amount', 0.5)
    width_scaling_factor = params.get('width_scaling_factor', 0.8)
    depth_scaling_factor = params.get('depth_scaling_factor', 0.8)

    # Choose the dataset and model
    dataset = 'cifar10'  # Change this to 'gtsrb' or 'self_collected_traffic_light' as needed
    model_type = 'AlexNet'  # Change this to the desired model type

    if dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders()
    elif dataset == 'gtsrb':
        train_loader, test_loader = get_gtsrb_loaders()
    elif dataset == 'self_collected_traffic_light':
        train_loader, test_loader = get_self_collected_loaders()
    else:
        raise ValueError("Invalid dataset")

    # Initialize model
    if model_type == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif model_type == 'ResNet18':
        model = get_resnet18(num_classes=10)
    elif model_type == 'ResNet34':
        model = get_resnet34(num_classes=10)
    elif model_type == 'VGG11':
        model = get_vgg11(num_classes=10)
    elif model_type == 'VGG13':
        model = get_vgg13(num_classes=10)
    elif model_type == 'VGG16':
        model = get_vgg16(num_classes=10)
    elif model_type == 'VGG19':
        model = get_vgg19(num_classes=10)
    elif model_type == 'TinyYOLO':
        model = TinyYOLO(num_classes=20)  # Adjust the number of classes for TinyYOLO
    else:
        raise ValueError("Invalid model type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Apply pruning
    prune_model(model, pruning_amount)

    # Apply weight sharing
    model = weight_sharing(model)

    # Apply width scaling
    model = width_scaling(model, width_scaling_factor)

    # Apply depth scaling
    model = depth_scaling(model, depth_scaling_factor)

    # Track time
    start_time = time.time()

    # Train and test the model
    for epoch in range(10):
        train(model, train_loader, criterion, optimizer, epoch)
        
        # Test the model on the test dataset
        test_loss, accuracy = test(model, test_loader, criterion)
        
        # Report intermediate result to NNI using test dataset accuracy
        nni.report_intermediate_result({"accuracy": accuracy, "time": time.time() - start_time})

    # Final results using test dataset
    final_time = time.time() - start_time
    nni.report_final_result({"accuracy": accuracy, "time": final_time})

if __name__ == "__main__":
    main()
