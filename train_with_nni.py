import torch
import torch.nn as nn
import torch.optim as optim
from training_evaluation import train, test
from data_preparation import get_cifar10_loaders, get_gtsrb_loaders
from models.AlexNet.model_alexnet import AlexNet
from models.ResNet.model_resnet import get_resnet18, get_resnet34
from models.VGG.model_vgg import get_vgg11, get_vgg13, get_vgg16, get_vgg19
from models.TinyYOLO.model_tiny_yolo import TinyYOLO
import torch.nn.utils.prune as prune
import time
import itertools
import pandas as pd
import os


def prune_model(model, amount):
    parameters_to_prune = (
        []
    )  # List of tuples of the form (module, name) where module is the module to prune and name is the parameter to prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )  # Prune the model


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
    # Define parameter ranges with more values to generate more model variants
    learning_rates = [
        0.001,
        0.003,
        0.005,
        0.007,
        0.009,
        0.01,
        0.03,
        0.05,
        0.07,
        0.09,
        0.1,
    ]
    pruning_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    width_scaling_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    depth_scaling_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Choose the dataset and model
    dataset = "cifar10"  # Change this to 'gtsrb' as needed
    model_type = "AlexNet"  # Change this to the desired model type

    if dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders()
    elif dataset == "gtsrb":
        train_loader, test_loader = get_gtsrb_loaders()
    else:
        raise ValueError("Invalid dataset")

    results_path = "model_variants_results.csv"
    if not os.path.exists(results_path):
        df = pd.DataFrame(
            columns=[
                "learning_rate",
                "pruning_amount",
                "width_scaling_factor",
                "depth_scaling_factor",
                "accuracy",
                "training_time",
                "test_time",
                "epoch",
            ]
        )
        df.to_csv(results_path, index=False)

    # Iterate over combinations of parameters to generate 100 variants
    param_combinations = list(
        itertools.product(
            learning_rates,
            pruning_amounts,
            width_scaling_factors,
            depth_scaling_factors,
        )
    )
    param_combinations = param_combinations[:100]  # Limit to 100 combinations

    for (
        learning_rate,
        pruning_amount,
        width_scaling_factor,
        depth_scaling_factor,
    ) in param_combinations:

        # Initialize model
        if model_type == "AlexNet":
            model = AlexNet(num_classes=10)
        elif model_type == "ResNet18":
            model = get_resnet18(num_classes=10)
        elif model_type == "ResNet34":
            model = get_resnet34(num_classes=10)
        elif model_type == "VGG11":
            model = get_vgg11(num_classes=10)
        elif model_type == "VGG13":
            model = get_vgg13(num_classes=10)
        elif model_type == "VGG16":
            model = get_vgg16(num_classes=10)
        elif model_type == "VGG19":
            model = get_vgg19(num_classes=10)
        elif model_type == "TinyYOLO":
            model = TinyYOLO(
                num_classes=20
            )  # Adjust the number of classes for TinyYOLO
        else:
            raise ValueError("Invalid model type")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.device = device

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
            train(model, train_loader, criterion, optimizer, epoch, device)
            test_start_time = time.time()
            test_loss, accuracy = test(model, test_loader, criterion, device)
            test_time = time.time() - test_start_time
            print(
                f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Test Time: {test_time:.2f} seconds"
            )

            # Capture time after testing
            current_time = time.time() - start_time

            # Append current results to CSV
            df = pd.DataFrame(
                [
                    {
                        "learning_rate": learning_rate,
                        "pruning_amount": pruning_amount,
                        "width_scaling_factor": width_scaling_factor,
                        "depth_scaling_factor": depth_scaling_factor,
                        "accuracy": accuracy,
                        "training_time": current_time,
                        "test_time": test_time,
                        "epoch": epoch + 1,
                    }
                ]
            )
            df.to_csv(results_path, mode="a", header=False, index=False)

            print(f"Epoch {epoch + 1} results saved.")


if __name__ == "__main__":
    main()
