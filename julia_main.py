# meine Main

from jointvae.models import VAE
from jointvae.training import Trainer
from torch import optim
from utils.dataloaders import get_arabic_dataloader
import torch
import numpy as np
from viz.visualize import Visualizer
import json
from chart_view import chart_viewer
import os

# For colab
#path_train = "/content/joint-vae/data/HandwrittenArabic/Train Images 13440x32x32/train"

# For testing in home directory
path_train = "data/HandwrittenArabic/Train Images 13440x32x32/train"
save_dir = "trained_models/arabic/model_disc28/"
test = '/content/joint-vae/trained_models/arabic/model.pt'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

train_loader = get_arabic_dataloader(path_to_data=path_train)

latent_spec = {'cont': 10, 'disc': [28]}
model = VAE(latent_spec=latent_spec, img_size=(1, 32, 32))
optimizer = optim.Adam(model.parameters(), lr=5e-4)
cont_capacity = [0.0, 5.0, 25000, 30.0]
disc_capacity = [0.0, 5.0, 25000, 30.0]
epochs = 1

trainer = Trainer(model, optimizer,
                  cont_capacity=cont_capacity,
                  disc_capacity=disc_capacity)

viz = Visualizer(model)

trainer.train(train_loader, epochs=epochs, save_training_gif=(save_dir + 'training.gif', viz))

torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

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

with open(save_dir + 'specs.json', 'w') as file:
    json.dump(specs, file)

# Show Chart of Loss
# x-Achse: Epochenanzahl
xdata = np.arange(epochs)
# y-Achse: loss des trainers (in training.py)
ydata = trainer.loss_arr
# chart_viewer(title, xlabel, ylabel, data for x-achse, data for y-achse)
# wird gespeichert unter title
chart_viewer(save_dir, "Loss Visualization (MNIST, SGD)", "Epochs", "Loss", xdata, ydata)

