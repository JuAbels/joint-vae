from utils.load_model import load
from viz.visualize import Visualizer as Viz
import matplotlib.pyplot as plt

path_to_model_folder = '/content/joint-vae/trained_models/arabic/'
model = load(path_to_model_folder)

# Create a Visualizer for the model
viz = Viz(model)
viz.save_images = False

samples = viz.samples()
plt.imsave("test.png", samples.numpy()[0, :, :])
