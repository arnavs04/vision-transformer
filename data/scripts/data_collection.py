import os
from pathlib import Path
import torchvision
import torchvision.transforms as transforms

# Setup path to data folder
data_path = Path("vision-transformer/data/")
image_path = data_path / "cifar-10"

# If the image folder doesn't exist, create it...
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download and prepare the CIFAR-10 dataset
print("Downloading CIFAR-10 data...")
trainset = torchvision.datasets.CIFAR10(root=image_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=image_path, train=False, download=True, transform=transform)

print("CIFAR-10 data downloaded and prepared.")
