# Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data.distributed
import argparse
import logging
import os
import sys
from PIL import ImageFile

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    """Test the model and log results."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc:.2f}%, Testing Loss: {total_loss:.4f}")
    print(f"Testing Accuracy: {100*total_acc:.2f}%, Testing Loss: {total_loss:.4f}")

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    """Train the model with validation."""
    best_loss = float('inf')
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects / len(train_loader.dataset)
        logger.info(f"Training Loss: {train_loss:.4f}, Accuracy: {100*train_acc:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss /= len(valid_loader.dataset)
        val_acc = val_corrects / len(valid_loader.dataset)
        logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {100*val_acc:.2f}%")
        
        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            logger.info("Best model saved.")

def net():
    """Initialize and return a ResNet50 model with custom layers for 5 classes."""
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all convolutional layers
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 5)  # 5 classes
    )
    return model

def create_data_loaders(data_dir, batch_size):
    logger.info("Creating data loaders")
    
    # Check if the directories exist
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


def save_model(model, model_dir):
    """Save the model state dictionary."""
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

def main(args):
    """Main execution function."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    # Train the model
    train(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs)

    # Test the model
    test(model, test_loader, loss_criterion, device)

    # Save the trained model
    save_model(model, args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Path to training data")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Path to save the model")

    args = parser.parse_args()
    main(args)
