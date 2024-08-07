import torch
import numpy as np
import matplotlib.pyplot as plt
from airfoil_dataset import AirfoilDataset
from simple_diffusion import Simple2DDiffusionModel  # Make sure to import the model definition

def plot_airfoil(airfoil, title="Airfoil"):
    plt.plot(airfoil[:, 0], airfoil[:, 1])
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.show()

def generate_and_plot_samples(model, num_samples=5, num_points=200):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            noisy_sample = torch.randn((1, 1, num_points, 2))  # Generate random noise
            generated_sample = model(noisy_sample).squeeze().numpy()  # Denoise using the model
            plot_airfoil(generated_sample, title=f"Generated Airfoil {i+1}")

if __name__ == '__main__':
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    
    # Initialize the model
    model = Simple2DDiffusionModel()
    model.load_state_dict(torch.load('simple_2d_diffusion_model.pth'))
    
    # Generate and plot sample airfoils
    generate_and_plot_samples(model, num_samples=5, num_points=200)
