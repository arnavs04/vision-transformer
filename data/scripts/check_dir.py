import os

directory = "data/cifar10/train"

if os.path.exists(directory):
    print("Exists")
else:
    print("Doesnt exist")