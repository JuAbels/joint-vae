# meine Main

import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from torch import optim
from utils.dataloaders import get_mnist_dataloaders


train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
