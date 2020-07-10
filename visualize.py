from utils.load_model import load
from viz.visualize import Visualizer as Viz
import matplotlib.pyplot as plt
from utils.dataloaders import get_arabic_dataloader
from utils.dataloaders import get_mnist_dataloaders


#path_to_model_folder = '/content/joint-vae/trained_models/arabic/'
path_to_model_folder = 'trained_models/arabic_2/'

model = load(path_to_model_folder)
#model = load("./trained_models/mnist/")

# Create a Visualizer for the model
viz = Viz(model)
viz.save_images = False

samples = viz.samples()
#plt.imsave("test.png", samples.numpy()[0, :, :])

dataloader = get_arabic_dataloader(batch_size=32, path_to_data="data/HandwrittenArabic/Test Images 3360x32x32/test")
#_, dataloader = get_mnist_dataloaders(batch_size=32)

first = []
for batch, labels in dataloader:
    break
    #if idx == 0:
    #    break

# torch.Size([32, 1, 32, 32])
recon = viz.reconstructions(batch, size=(8, 8))

plt.savefig


