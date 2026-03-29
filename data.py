"""Split MNIST dataset creation for continual learning experiments.

Creates 5 binary classification tasks: 0v1, 2v3, 4v5, 6v7, 8v9.
Each task relabels its two digit classes to 0 and 1.
"""

import ssl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# Workaround for SSL certificate issues when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def get_split_mnist(data_dir="./data", batch_size=256):
    """Load Split MNIST as a list of (train_loader, test_loader, (class_a, class_b))."""
    mnist_train = datasets.MNIST(data_dir, train=True, download=True)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True)

    mean, std = 0.1307, 0.3081
    all_train_x = (mnist_train.data.float().view(-1, 784) / 255.0 - mean) / std
    all_test_x = (mnist_test.data.float().view(-1, 784) / 255.0 - mean) / std

    tasks = []
    for t in range(5):
        ca, cb = t * 2, t * 2 + 1

        tr_mask = (mnist_train.targets == ca) | (mnist_train.targets == cb)
        te_mask = (mnist_test.targets == ca) | (mnist_test.targets == cb)

        tr_loader = DataLoader(
            TensorDataset(all_train_x[tr_mask], (mnist_train.targets[tr_mask] == cb).long()),
            batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(
            TensorDataset(all_test_x[te_mask], (mnist_test.targets[te_mask] == cb).long()),
            batch_size=batch_size)
        tasks.append((tr_loader, te_loader, (ca, cb)))

    return tasks
