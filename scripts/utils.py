"""
Contains various utility functions for PyTorch model building, training and saving.
"""

import torch
from torch import nn
import numpy as np
from pathlib import Path
import math
import random

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def number_of_patches(height: int, 
                      width: int, 
                      patch_size: int):
    """Returns the number of patches for a given image size and patch size.

    Args:
        height (int): The height of the input image.
        width (int): The width of the input image.
        patch_size (int): The size of each patch (assuming square patches).

    Returns:
        int: The number of patches that can be obtained from the input image.

    Example usage:
        num_patches = number_of_patches(height=256, width=256, patch_size=16)
        print(num_patches)  # Output: 256
    """
    return int((height * width) / patch_size**2)

def embedding_output_shape(height: int, 
                           width: int, 
                           patch_size: int, 
                           channels: int):
    """Calculates the output shape of an embedding layer based on input image dimensions, patch size, and number of channels.

    Args:
        height (int): The height of the input image.
        width (int): The width of the input image.
        patch_size (int): The size of each patch (assuming square patches).
        channels (int): The number of channels in the input image.

    Returns:
        tuple: A tuple representing the output shape (number of patches, patch size squared, number of channels).

    Example usage:
        output_shape = embedding_output_shape(height=256, width=256, patch_size=16, channels=3)
        print(output_shape)  # Output: (256, 256, 3)
    """
    return (number_of_patches(height, width, patch_size), patch_size**2, channels)

def is_torch_available():
    """
    Checks if the PyTorch library is available in the current environment.

    This function attempts to import the PyTorch library and returns True if the import is successful,
    indicating that PyTorch is available. If the import fails (ImportError is raised), it returns False.

    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        return True
    except ImportError:
        return False

    
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def count_parameters(model: nn.Module):
    """
    Calculate the total number of trainable parameters in a PyTorch model.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super(MyModel, self).__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...         self.conv = nn.Conv2d(3, 6, 3)
        ...     def forward(self, x):
        ...         return self.conv(self.linear(x))
        >>> model = MyModel()
        >>> count_parameters(model)
        66
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   