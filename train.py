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
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

def weight_sharing(model):
    shared_weights = {}
    private_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight_shape = module.weight.shape
            if weight_shape in shared_weights:
                shared_weights[weight_shape].append(module.weight)
            else:
                shared_weights[weight_shape] = [module.weight]
            private_weights[name] = module.weight
    return shared_weights, private_weights

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

def save_model(model, model_type, learning_rate, pruning_amount, width_scaling_factor, depth_scaling_factor, shared_weights):
    # Create model directory if it doesn't exist
    model_dir = f"trained_models/{model_type}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Generate weight sharing information string
    #weight_sharing_info = "_".join([f"{k}_{len(v)}" for k, v in shared_weights.items()])

    # Construct the model path with weight sharing information
    model_path = os.path.join(
        model_dir,
        f"{model_type}_lr{learning_rate}_pa{pruning_amount}_wsf{width_scaling_factor}_dsf{depth_scaling_factor}.pth"
    )

    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save shared weights information
    shared_weights_path = os.path.join(model_dir, f"{model_type}_shared_weights.pth")
    torch.save(shared_weights, shared_weights_path)
    print(f"Shared weights saved to {shared_weights_path}")

def main():
    # Define parameter ranges with more values to generate more model variants
    learning_rate = 0.01  # Default learning rate
    pruning_amounts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    width_scaling_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    depth_scaling_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Choose the dataset and model
    dataset = 'cifar10'  # Change this to 'gtsrb' as needed
    model_type = 'ResNet18'  # Change this to the desired model type

    if dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders()
    elif dataset == 'gtsrb':
        train_loader, test_loader = get_gtsrb_loaders()
    else:
        raise ValueError("Invalid dataset")

    results_path = 'model_variants_results.csv'
    if not os.path.exists(results_path):
        df = pd.DataFrame(columns=["learning_rate", "pruning_amount", "width_scaling_factor", 
                                   "depth_scaling_factor", "accuracy", "training_time", "test_time", "epoch"])
        df.to_csv(results_path, index=False)

    # Iterate over combinations of parameters to generate 100 variants
    param_combinations = list(itertools.product(pruning_amounts, width_scaling_factors, depth_scaling_factors))
    param_combinations = param_combinations[:100]  # Limit to 100 combinations

    for pruning_amount, width_scaling_factor, depth_scaling_factor in param_combinations:

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
        model.device = device

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Apply pruning
        prune_model(model, pruning_amount)

        # Apply weight sharing
        shared_weights, private_weights = weight_sharing(model)

        # Apply width scaling
        model = width_scaling(model, width_scaling_factor)

        # Apply depth scaling
        model = depth_scaling(model, depth_scaling_factor)

        # Track time
        start_time = time.time()



        # Initialize the best accuracy and best epoch tracking
        best_accuracy = 0.0
        best_epoch_data = None  # To store the data of the best accuracy epoch
        loss_threshold = 1e-7  # Define a threshold for loss to decide when to stop training

        # Train and test the model
        for epoch in range(100):
            train(model, train_loader, criterion, optimizer, epoch, device)
            
            test_loss, accuracy = test(model, test_loader, criterion, device)
            print(f"Epoch {epoch + 1}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Stop training if loss is below the threshold
            if test_loss <= loss_threshold:
                print(f"Stopping training early as loss {test_loss:.4f} is below the threshold {loss_threshold}.")
                break

            # Adjust learning rate
            scheduler.step(test_loss)

            # Print current learning rate
            current_lr = scheduler._last_lr[0]
            print(f"Current Learning Rate: {current_lr}")

            # Update best accuracy if the current accuracy is higher than the best found so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch_data = {
                    "learning_rate": current_lr,
                    "pruning_amount": pruning_amount,
                    "width_scaling_factor": width_scaling_factor,
                    "depth_scaling_factor": depth_scaling_factor,
                    "accuracy": best_accuracy,
                    "training_time": time.time() - start_time,
                    "epoch": epoch + 1
                }

        # After training, calculate average test time
        if best_epoch_data:
            total_test_time = 0
            num_tests = 10
            for _ in range(num_tests):
                start_test_time = time.time()
                test(model, test_loader, criterion, device)
                total_test_time += time.time() - start_test_time
            average_test_time = total_test_time / num_tests

            # Append the average test time to the best epoch data
            best_epoch_data["average_test_time"] = average_test_time

            # Save best accuracy epoch data to CSV
            df = pd.DataFrame([best_epoch_data])
            results_path = 'best_accuracy_epoch_results.csv'  # Specify the correct path
            df.to_csv(results_path, mode='w', header=True, index=False)
            print(f"Best accuracy epoch results saved: Epoch {best_epoch_data['epoch']} with Accuracy: {best_accuracy:.2f}% and Average Test Time: {average_test_time:.2f} seconds")

            # Save the model with the best accuracy information and average test time
            save_model(model, model_type, best_epoch_data['learning_rate'], best_epoch_data['pruning_amount'],
                    best_epoch_data['width_scaling_factor'], best_epoch_data['depth_scaling_factor'],
                    shared_weights, average_test_time)

if __name__ == "__main__":
    main()
