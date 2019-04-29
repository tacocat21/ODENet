import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def get_cifar_10(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def get_stl_10(data_aug=False, batch_size=128, test_batch_size=1000):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(234),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.STL10(root='.data/stl10', split='train', download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.STL10(root='.data/stl10', split='train', download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.STL10(root='.data/stl10', split='test', download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    dataset = datasets.STL10(root='.data/stl10', split='train', download=True, transform=transform)
    print(get_mean_and_std(dataset))