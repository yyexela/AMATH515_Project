###############################
# Imports # Imports # Imports #
###############################

import numpy as np
from torch import permute, float64
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pathlib
import os

from amath515_pkg.src.helpers import get_config

#########################################
# Dataset Functions # Dataset Functions #
#########################################

def get_cifar10_datasets(batch_size = None, shuffle = None):
    """
    Returns the training and testing datasets for CIFAR10
    """
    # Load config file and values
    config = get_config()
    if batch_size is None:
        batch_size = config["CIFAR10_batch_size"]
    if shuffle is None:
        shuffle = config["CIFAR10_shuffle"]

    # Load dataset
    # Download and load dataset
    base_dir = pathlib.Path().resolve() # get current file path
    dataset_dir = os.path.join(base_dir, "Datasets") # where to save dataset

    training_data = datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dl = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_dl, test_dl

def get_cifar10_wrapped_datasets(device=None):
    """
    Return the CIFAR10 training and testing datasets but optimized for training
    """
    train_dl, test_dl = get_cifar10_datasets()
    train_dl, test_dl = wrap_cifar10_datasets(train_dl, test_dl, device=device)

    return train_dl, test_dl

def wrap_cifar10_datasets(train_dl, test_dl, device=None):
    """
    Given CIFAR10 datasets, wrap them to optimize them for training
    """
    # Load config file and values
    config = get_config()
    if device is None:
        device = config["device"]

    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func

        def __len__(self):
            return len(self.dl)

        def __iter__(self):
            batches = iter(self.dl)
            for b in batches:
                # *b makes it so that the input to func is two variables: the images and the labels
                # yield is like return except it stops execution until the object is "grabbed"
                yield (self.func(*b)) 

    def preprocess(x, y):
        return x.to(device).to(float64), y.to(device)

    train_dl = WrappedDataLoader(train_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    return train_dl, test_dl

def get_cifar10_labels():
    """
    Mapping from number to string for CIFAR10 labels
    """
    cifar10_labels = ["airplane",
                      "automobile",
                      "bird",
                      "cat",
                      "deer",
                      "dog",
                      "frog",
                      "horse",
                      "ship",
                      "truck"]
    return cifar10_labels

#################################
# Data Analysis # Data Analysis #
#################################

def plot_cifar10_sample(train_dl = None):
    """
    Plot a 4x4 grid sapmle of CIFAR10 images
    """

    # Number of images we'll plot
    ncols = 4
    nrows = 4

    # Load dataset if not given
    if train_dl is None:
        train_dl, _ = get_cifar10_datasets(shuffle=False, batch_size=ncols*nrows)
    batch = next(iter(train_dl))
    cifar10_labels = get_cifar10_labels()

    # Create our figure as a grid
    fig1, f1_axes = plt.subplots(ncols=ncols, nrows=nrows, constrained_layout=True, figsize=(3.5*ncols,3.5*nrows), facecolor='white')
    fig1.set_dpi(50.0)

    # Plot the modified images
    for c in range(ncols):
        for r in range(nrows):
                image = batch[0][r*ncols+c]
                label_real = batch[1][r*ncols+c]
                image_show = permute(image, (1,2,0)) # Change ordering of dimensions

                ax = f1_axes[r][c]
                ax.set_axis_off()
                ax.set_title(f"{cifar10_labels[label_real]}", fontsize=20)
                ax.imshow(np.asarray(image_show), cmap='gray')

    plt.show()
