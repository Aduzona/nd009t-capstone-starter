# Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import logging
from PIL import ImageFile
from smdebug import modes
import smdebug.pytorch as smd

# Define the test function
def test(model, test_loader, criterion, hook, device):
    model.eval()
    hook.set_mode(modes.EVAL)
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Loss: {running_loss / len(test_loader.dataset)}, Accuracy: {accuracy}%")

# Define the train function with validation
def train(model, train_loader, val_loader, criterion, optimizer, hook, device, epochs):
    for epoch in range(epochs):
        model.train()
        hook.set_mode(modes.TRAIN)
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader.dataset)}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader.dataset)}, Accuracy: {val_accuracy}%")

# Simple model definition
def simple_model(): #net()
    base_model = models.resnet50(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False
    num_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        nn.Linear(256, 5)  # 5 classes
    )
    return base_model

# Data loader function
def create_data_loaders(data_dir, batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'validation'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)  # Updated here

    return train_loader, val_loader, test_loader

# Main function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simple_model().to(device)
    
    # Use CrossEntropyLoss for classification
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    # Debugging hook
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    # Data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size)
    
    # Train the model
    train(model, train_loader, val_loader, loss_criterion, optimizer, hook, device, args.epochs)
    
    # Test the model
    test(model, test_loader, loss_criterion, hook, device)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'simple_model.pth'))

# Argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # SageMaker environment parameters
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Directory containing the training data")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Directory to save the trained model")

    # New argument for test batch size
    parser.add_argument("--test-batch-size", type=int, default=64, help="Batch size for testing")
    
    args = parser.parse_args()
    
    main(args)
