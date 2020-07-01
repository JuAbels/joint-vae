# Jascha's Main

import torch
import numpy as np
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_mnist_dataloaders
from torch import optim
from chart_view import chart_viewer



batch_size = 64
lr = 5e-4
epochs = 10 # Tr: reduced epoch_nr

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader, _ = get_mnist_dataloaders(batch_size=batch_size)
img_size = (1, 32, 32)

# Define latent spec and model
latent_spec = {'cont': 10, 'disc': [10]}
model = VAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=[0.0, 5.0, 25000, 30],
                  disc_capacity=[0.0, 5.0, 25000, 30],
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), 'example-model.pt')

# Show Chart of Loss
xdata = np.arange(epochs)
ydata = trainer.loss_arr
chart_viewer("Loss Visualization (MNIST, ADAM)", "Epochs", "Loss", xdata, ydata)