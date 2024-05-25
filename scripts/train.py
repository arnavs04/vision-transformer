"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import torchvision
from torchvision import transforms
from pathlib import Path

import data_setup, engine, model_builder, utils

# Define paths
data_path = Path("./data/")
train_dir = data_path / "CIFAR10/train"
test_dir = data_path / "CIFAR10/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.003

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
    transforms.RandomRotation(degrees=15),   # Randomly rotate the image by +/- 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor with mean and std deviation
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    transform=data_transform,
    batch_size=BATCH_SIZE
)
