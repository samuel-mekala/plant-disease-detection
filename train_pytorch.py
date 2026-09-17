"""
Plant Disease Detection — High-Performance PyTorch Transfer Learning Training
Based on Senior Design Project Report (VIT-AP University)

Trains ResNet18 / MobileNetV3 on the 38-class Plant Diseases Dataset.
Saves trained weights to 'plant_disease_model.pth' for real-time inference in app.py.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_data_dirs():
    candidates = [
        "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
        "../input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return os.path.join(c, "train"), os.path.join(c, "valid")
    return candidates[0] + "/train", candidates[0] + "/valid"

def train_model(epochs=3, batch_size=64, lr=0.001):
    train_dir, valid_dir = get_data_dirs()
    if not os.path.exists(train_dir):
        print(f"Dataset path '{train_dir}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Image Transforms matching ImageNet & Report Specifications
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(train_dataset)} training images across {num_classes} classes.")
    print(f"Loaded {len(valid_dataset)} validation images.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Transfer Learning with Pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True  # Fine-tune all layers for maximum accuracy

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

            if (idx + 1) % 100 == 0:
                print(f"Batch {idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_corrects.double() / val_total
        print(f"Valid Loss: {val_epoch_loss:.4f} | Valid Acc: {val_epoch_acc*100:.2f}%")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'accuracy': float(val_epoch_acc)
            }, "plant_disease_model.pth")
            print(f"--> Saved best checkpoint (Acc: {val_epoch_acc*100:.2f}%) to 'plant_disease_model.pth'")

    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed//60:.0f}m {elapsed%60:.0f}s with Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_model(epochs=3, batch_size=64, lr=0.001)
