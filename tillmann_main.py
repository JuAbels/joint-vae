# Tillmanns Main

import torch
import numpy as np
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_mnist_dataloaders
from torch import optim

# Own coded function imports
from chart_view import chart_viewer
import json

batch_size = 64
lr = 5e-4
epochs = 1

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader, _ = get_mnist_dataloaders(batch_size=batch_size)
img_size = (1, 32, 32)

# Define latent spec and model
latent_spec = {'cont': 10, 'disc': [10]}
model = VAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda, loss="cauchy")
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=[0.0, 5.0, 25000, 30],
                  disc_capacity=[0.0, 5.0, 25000, 30],
                  use_cuda=use_cuda)

# Train model for given number of epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), 'example-model.pt')

specs = {"cont_capacity": trainer.cont_capacity,
         "disc_capacity": trainer.disc_capacity,
         "record_loss_every": trainer.record_loss_every,
         "batch_size": trainer.batch_size,
         "latent_spec": model.latent_spec,
         "epochs": [epochs],
         "experiment_name": "2",
         "lr": [0.0005],
         "print_loss_every": trainer.print_loss_every,
         "dataset": "arabicLetter"}

with open('specs.json', 'w') as file:
    json.dump(specs, file)

# Show Chart of Loss
# x-Achse: Epochenanzahl
xdata = np.arange(epochs)
# y-Achse: loss des trainers (in training.py)
ydata = trainer.loss_arr
# chart_viewer(title, xlabel, ylabel, data for x-achse, data for y-achse)
# wird gespeichert unter title
chart_viewer("Loss Visualization (MNIST, cauchy)", "Epochs", "Loss", xdata, ydata)