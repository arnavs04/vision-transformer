import os

directory = "data/cifar-10/train"

if os.path.exists(directory):
    print("Exists")
else:
    print("Doesnt exist")