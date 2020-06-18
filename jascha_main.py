from jointvae.models import VAE
from jointvae.training import Trainer
from torch.optim import Adam
from viz.visualize import Visualizer as Viz

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

# Build a trainer and train model
optimizer = Adam(model.parameters())

trainer = Trainer(model, optimizer,
                  cont_capacity=[0., 5., 25000, 30.],
                  disc_capacity=[0., 5., 25000, 30.])
trainer.train(dataloader, epochs=10)

# Visualize samples from the model
viz = Viz(model)
samples = viz.samples()

# Do all sorts of fun things with model
