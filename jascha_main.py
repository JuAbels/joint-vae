import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_mnist_dataloaders
from torch import optim
from viz.visualize import Visualizer as Viz
from utils.load_model import load

model = load('joint-vae/trained_models/chairs/')

# Visualize samples from the model
viz = Viz(model)
samples = viz.samples(filename='test.png')
