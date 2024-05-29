import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class Datasets():
    def get_data(data="MNIST", batch_size=128):
        if data == "MNIST":
            train_dataset = datasets.MNIST(
                root="./mnist_data/",
                train=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
                ),
                download=True,
            )
            test_dataset = datasets.MNIST(
                root="./mnist_data/",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
                ),
                download=False,
            )
        elif data == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(
                root="./mnist_data/",
                train=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
                ),
                download=True,
            )
            test_dataset = datasets.FashionMNIST(
                root="./mnist_data/",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
                ),
                download=False,
            )
            
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader