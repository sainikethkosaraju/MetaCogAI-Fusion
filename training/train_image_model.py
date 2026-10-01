# train_image_model.py

# ================================
# MetaCogAI - Image Classifier Training Script
# Trains the image expert (ResNet-18, from scratch) on CIFAR-10.
# Run from the repository root:  python training/train_image_model.py
# ================================

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "weights/image_resnet18_cifar10.pth"

# -------------------------
# Device Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Transforms & Data Loaders
# -------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# Model, Loss, Optimizer
# -------------------------
model = models.resnet18(weights=None, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# Training Loop
# -------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training complete.")

# -------------------------
# Save Model
# -------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
