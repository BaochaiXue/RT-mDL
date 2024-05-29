import torch
import torch.nn as nn

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # print every 100 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def global_train(shared_weights, optimizer, criterion, train_loader, device, model):
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for weight_list in shared_weights.values():
            for weight in weight_list:
                optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def local_train(private_weights, optimizer, criterion, train_loader, device, model):
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for name, weight in private_weights.items():
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
