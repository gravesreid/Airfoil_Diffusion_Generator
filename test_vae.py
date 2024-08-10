import torch
import os
from torch.utils.data import DataLoader
from airfoil_dataset_1d import AirfoilDataset
import matplotlib.pyplot as plt
from utils_1d import *
from vae import *

airfoil_dim = 200
latent_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = VAE(airfoil_dim, latent_dim).to(device)
model_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

# Initialize dataset and dataloader
airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

airfoil_x = dataset.get_x()

def plot_airfoils_grid(original_airfoils, reconstructed_airfoils, airfoil_x, grid_size=4):
    num_airfoils = original_airfoils.shape[0]
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    
    for i in range(num_airfoils):
        row = i // grid_size
        col = i % grid_size

        # Plot original airfoil
        ax = axs[row, col]
        original = original_airfoils[i].detach().cpu().numpy()
        ax.scatter(airfoil_x, original, color='black') 
        reconstructed = reconstructed_airfoils[i].detach().cpu().numpy()
        ax.scatter(airfoil_x, reconstructed, color='red')
        ax.set_title(f'Original Airfoil {i+1}')
        ax.set_aspect('equal')
        ax.axis('off')

        # Plot reconstructed airfoil
       # ax = axs[row * 2 + 1, col]
       # reconstructed = reconstructed_airfoils[i].detach().cpu().numpy()
       # ax.scatter(airfoil_x, reconstructed, color='red')
       # ax.set_title(f'Reconstructed Airfoil {i+1}')
       # ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()



# plot a 4x4 grid of all the airfoils
original_airfoils = []
reconstructed_airfoils = []
for i, data in enumerate(dataloader):
    airfoil = data['train_coords_y'].float().to(device)
    reconstructed_airfoil, _, _ = model(airfoil)
    reconstructed_airfoil = reconstructed_airfoil.detach().cpu().numpy()
    airfoil = airfoil.detach().cpu().numpy()
    original_airfoils.append(airfoil)
    reconstructed_airfoils.append(reconstructed_airfoil)


    
for data in dataloader:
    airfoil = data['train_coords_y'].float().to(device)
    reconstructed_airfoil, _, _ = model(airfoil)

    plot_airfoils_grid(airfoil, reconstructed_airfoil, airfoil_x)
