import torchvision
import torch

transforms = torchvision.transforms

train_data = torchvision.datasets.CIFAR10(root='./data',
                                        train=True, 
                                        download=True)

test_data = torchvision.datasets.CIFAR10(root='./data', 
                                        train=False, 
                                        download=True)