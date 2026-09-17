"""
Plant Disease Detection — High-Performance PyTorch Transfer Learning Training
Based on Senior Design Project Report (VIT-AP University)

Trains ResNet-18 on the updated Plant Diseases Dataset in 'data/'.
Automatically splits data into train/valid subsets and saves trained weights to 'plant_disease_model.pth'.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

def get_dataset_root():
    candidates = [
        "data",
        "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)",
    ]
    for c in candidates:
        if os.path.exists(c):
            subdirs = [os.path.join(c, d) for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if any(any(f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG')) for f in os.listdir(sd)) for sd in subdirs if os.path.isdir(sd)):
                return c
            if os.path.exists(os.path.join(c, "train")):
                return os.path.join(c, "train")
    return "data"

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train_model(epochs=4, batch_size=64, lr=0.0003):
    data_dir = get_dataset_root()
    print(f"Using dataset directory: '{data_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    # Advanced In-The-Wild Domain Augmentations (Handles outdoor lighting, background noise, angled shots)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset with separate train vs val transforms
    dataset_raw = datasets.ImageFolder(data_dir)
    class_names = dataset_raw.classes
    num_classes = len(class_names)

    print(f"Loaded {len(dataset_raw)} total images across {num_classes} classes:")
    for idx, cname in enumerate(class_names):
        print(f"  [{idx:02d}] {cname}")

    val_size = int(0.2 * len(dataset_raw))
    train_size = len(dataset_raw) - val_size
    train_subset, val_subset = random_split(dataset_raw, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = TransformedDataset(train_subset, train_transform)
    val_dataset = TransformedDataset(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # ResNet-18 with Dropout & Fine-Tuning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    
    # Custom head with Dropout regularization
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    best_acc = 0.0
    model_path = "plant_disease_model.pth"
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                best_acc = ckpt.get('accuracy', 0.0)
                print(f"🔄 Resumed from saved checkpoint '{model_path}' (Previous Best Acc: {best_acc*100:.2f}%)", flush=True)
        except Exception as e:
            print(f"Starting fresh training (notice: {e})", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n================ Epoch {epoch+1}/{epochs} ================", flush=True)
        
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
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

            if (idx + 1) % 50 == 0:
                acc_batch = (torch.sum(preds == labels.data).float().item() / inputs.size(0)) * 100
                print(f" Batch {idx+1:03d}/{len(train_loader)} | Loss: {loss.item():.4f} | Batch Acc: {acc_batch:.1f}%", flush=True)

        scheduler.step()
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        print(f"--> Epoch {epoch+1} Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%", flush=True)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)

        val_epoch_acc = val_corrects / val_total
        print(f"--> Epoch {epoch+1} Valid Loss: {(val_loss/val_total):.4f} | Valid Acc: {val_epoch_acc*100:.2f}%", flush=True)

        if val_epoch_acc >= best_acc:
            best_acc = val_epoch_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'accuracy': float(val_epoch_acc)
            }, "plant_disease_model.pth")
            print(f"🏆 Saved Best Model Checkpoint to 'plant_disease_model.pth' (Acc: {val_epoch_acc*100:.2f}%)", flush=True)

    elapsed = time.time() - start_time
    print(f"\n✅ Training Complete in {elapsed//60:.0f}m {elapsed%60:.0f}s! Best Validation Accuracy: {best_acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    train_model(epochs=4, batch_size=64, lr=0.0003)

