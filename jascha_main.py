# meine Main

import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from torch import optim
from utils.dataloaders import get_mnist_dataloaders
from utils.dataloaders import get_arabic_dataloader
import matplotlib.pyplot as plt
import helper
import torch
from torchvision import datasets, transforms, utils
import torchvision
import numpy as np
from viz.visualize import Visualizer

path_csv = "/Users/juliaabels/Workspace/Uni/SS20/Deep Learning/project/data/HandwrittenArabic/" \
           "Arabic Handwritten Characters Dataset CSV/csvTrainImages 13440x1024.csv"

path_train = "/Users/juliaabels/Workspace/Uni/SS20/Deep Learning/project/data/HandwrittenArabic" \
             "/Train Images 13440x32x32/train"


#train_loader = get_arabic_dataloader(csv_file=path_csv,
#                                      batch_size=64,
#                                      path_to_data="/Users/juliaabels/Workspace/Uni/SS20/Deep Learning/
#                                      project/data/HandwrittenArabic/Arabic Handwritten Characters Dataset CSV")

train_loader = get_arabic_dataloader(path_to_data=path_train)
# train_loader, test_loader = get_mnist_dataloaders(batch_size=64)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# dataiter = iter(train_loader)
# images = dataiter.next()
# plt.figure(figsize=(16,8))
# imshow(torchvision.utils.make_grid(images))
print("test")

latent_spec = {'cont': 10, 'disc': [10]}
model = VAE(latent_spec=latent_spec, img_size=(1, 32, 32))
optimizer = optim.Adam(model.parameters(), lr=5e-4)
cont_capacity = [0.0, 5.0, 25000, 30.0]
disc_capacity = [0.0, 5.0, 25000, 30.0]

trainer = Trainer(model, optimizer,
                  cont_capacity=cont_capacity,
                  disc_capacity=disc_capacity)

viz = Visualizer(model)

trainer.train(train_loader, epochs=10, save_training_gif=('./training.gif', viz))
