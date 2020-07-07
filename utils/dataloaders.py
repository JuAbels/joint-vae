import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os
import torch
import os.path as osp
from PIL import Image


def get_mnist_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='../fashion_data'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../dsprites-data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64'):
    """CelebA dataloader with (64, 64) images."""
    celeba_data = CelebADataset(path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=True)
    return celeba_loader


def get_arabic_dataloader(batch_size=64, path_to_data='../data/ArabicLetter/csvTrainImages 13440x1024.csv'):
    """Handwritten arabic dataloader with (32, 32) images."""
    # arabic_data = ArabicDataset(csv_file, path_to_data,
    #                            transform=transforms.ToTensor())
    arabic_data = ArabicDataset(path_to_data)
    arabic_loader = DataLoader(arabic_data, batch_size=batch_size,
                               shuffle=True)
    return arabic_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class ArabicDataset(Dataset):
    """Handwritten arabic letter dataset with 32 by 32 images."""

    def __init__(self, root):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = root
        self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(root, '*.png'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)

    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        self.label = int(os.path.splitext(os.path.basename(self.filenames[index]))[0].rsplit("_", 1)[-1])
        return self.transform(image), self.label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
