import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from airfoil_dataset_1d import AirfoilDataset
from torch.utils.data import DataLoader
from vae import VAE
from LucidDiffusion import *
import aerosandbox as asb
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

vae_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
diffusion_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_cd_run_1/best_model.pt"

# function to smooth out y coordinates
def smooth_y_coords(y_coords, method='moving_average', window_size=3, **kwargs):
    if method == 'moving_average':
        window = np.ones(window_size) / window_size
        pad = window_size // 2
        y_padded = np.pad(y_coords, (pad, pad), mode='edge')
        return np.convolve(y_padded, window, mode='valid')
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(y_coords, sigma=sigma)
    
    elif method == 'savitzky_golay':
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(y_coords, window_length=window_size, polyorder=polyorder)
    
    else:
        raise ValueError("Unknown smoothing method: choose from 'moving_average', 'gaussian', or 'savitzky_golay'.")



# Parameters
airfoil_dim = 200
latent_dim = 100
n_samples = 16
unet_dim = 12
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the trained VAE model
vae_model = VAE(airfoil_dim, latent_dim).to(device)
vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
vae_model.eval()

# Load the trained diffusion model
diffusion_model = Unet1DConditional(unet_dim, cond_dim=2, channels=2, dim_mults=(1,2,4)).to(device)
diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
diffusion_model.eval()
diffusion = GaussianDiffusion1D(diffusion_model, seq_length=latent_dim).to(device)

# Initialize dataset
airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

airfoil_x = dataset.get_x()

# Generate latent vectors with diffusion
# make conditioning tensor of shape (n_samples, 1)
# Define ranges for CL and CD
cl_range = np.linspace(-0.2, 2, num=15)
cd_range = np.linspace(0.0, 0.01, num=8)

# Create a grid of CL and CD values
cl_cd_grid = np.array([[cl, cd] for cl in cl_range for cd in cd_range])
n_samples = cl_cd_grid.shape[0]

# Generate airfoils for each CL, CD pair
latent_vectors = []
cl_values = []
cd_values = []

samples_generated = 0
batch_size = 20  # Set the batch size

for conditioning_values in cl_cd_grid:
    conditioning_cl = torch.tensor([conditioning_values[0]], dtype=torch.float32).to(device).unsqueeze(1).repeat(batch_size, 1)
    conditioning_cd = torch.tensor([conditioning_values[1]], dtype=torch.float32).to(device).unsqueeze(1).repeat(batch_size, 1)
    combined_conditioning = torch.cat([conditioning_cl, conditioning_cd], dim=1)

    generated_y = diffusion.sample(batch_size=batch_size, conditioning=combined_conditioning)
    print(f'Generated_y shape: {generated_y.shape}')
    
    for i in range(batch_size):
        y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
        
        # Get the latent vector using the VAE encoder
        latent_vector = vae_model.enc(y_coords)
        latent_vector = latent_vector[0].detach().cpu().numpy()
        latent_vector = torch.tensor(latent_vector)
        print(f'Latent vector shape: {latent_vector.shape}')
        latent_vectors.append(latent_vector)
        
        # Store the conditioning values for each sample
        cl_values.append(conditioning_values[0])
        cd_values.append(conditioning_values[1])

    samples_generated += batch_size
    print(f"Generated airfoil batch: {samples_generated}/{n_samples * batch_size}")

latent_vectors = np.array(latent_vectors)

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2)
latent_tsne = tsne.fit_transform(latent_vectors)

# Plot t-SNE with colors based on conditioning values
plt.figure(figsize=(10, 6))
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cl_values, cmap='viridis', label='Generated Airfoils')
plt.colorbar(label='CL Value')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of Latent Vectors Colored by CL Value')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cd_values, cmap='plasma', label='Generated Airfoils')
plt.colorbar(label='CD Value')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of Latent Vectors Colored by CD Value')
plt.show()