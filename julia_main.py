# meine Main

from jointvae.models import VAE
from jointvae.training import Trainer
from torch import optim
from utils.dataloaders import get_arabic_dataloader, get_shape_dataloader
import torch
import numpy as np
from viz.visualize import Visualizer
import json
from chart_view import chart_viewer
import os

path_train = ''
save_dir = ''

# TODO correct path
dataset = "shape"
colab = False
model_name = "model_test_loss/"

if dataset == 'shape' and colab == False:
    path_train += "/Users/juliaabels/Workspace/Uni/SS20/Deep Learning/project/data/Shape/shapes"
    save_dir += "trained_models/arabic/" + model_name
elif dataset == 'shape' and colab == True:
    path_train += "/content/joint-vae/data/Shape/shapes"
    save_dir += "/content/joint-vae/trained_models/arabic/" + model_name
elif dataset == 'arabic' and colab == False:
    # For testing in home directory
    path_train += "data/HandwrittenArabic/Train Images 13440x32x32/train"
    save_dir += "trained_models/arabic/" + model_name
elif dataset == 'arabic' and colab == True:
    # For colab
    path_train += "/content/joint-vae/data/HandwrittenArabic/Train Images 13440x32x32/train"
    save_dir += "/content/joint-vae/trained_models/arabic/" + model_name

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

if dataset == 'arabic':
    train_loader = get_arabic_dataloader(path_to_data=path_train)
else:
    train_loader = get_shape_dataloader(path_to_data=path_train)


# TODO {1, 5, 8, 15}
latent_spec = {'cont': 15, 'disc': [28]}

model = VAE(latent_spec=latent_spec, img_size=(1, 32, 32))

# TODO Learning Rate
optimizer = optim.Adam(model.parameters(), lr=5e-4)
cont_capacity = [0.0, 5.0, 25000, 30.0]
disc_capacity = [0.0, 5.0, 25000, 30.0]

# TODO Epoch number 100
epochs = 1

# TODO
loss_dict = {"a": "cross_entro", "b": "cauchy", "c": "MSE"}
loss = loss_dict["a"]
trainer = Trainer(model, optimizer,
                  cont_capacity=cont_capacity,
                  disc_capacity=disc_capacity,
                  loss=loss)

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
         "lr": [0.0005],                                # TODO anpassen
         "print_loss_every": trainer.print_loss_every,
         "dataset": "arabicLetter"}
with open(save_dir + 'specs.json', 'w') as file:
    json.dump(specs, file)

# Show Chart of Loss
xdata = np.arange(epochs)
ydata = trainer.loss_arr

# TODO Loss
chart_viewer(save_dir, "Loss Visualization (Arabic, Adam, %s)" % loss, "Epochs", "Loss", xdata, ydata)

save_loss = np.array(trainer.loss_save).reshape(-1, 1)
np.savetxt(save_dir+"Loss.txt", save_loss)
np.savetxt(save_dir+"Mean_loss.txt", ydata.reshape(-1, 1))

