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
print(f"Using device: {device}")
conditioning_cl = torch.tensor([2.0]).to(device).unsqueeze(1)
conditioning_cd = torch.tensor([.02]).to(device).unsqueeze(1)
combined_conditioning = torch.cat([conditioning_cl, conditioning_cd], dim=1)
print(f'conditioning shape: {conditioning_cl.shape}')

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
combined_conditioning = combined_conditioning.repeat(n_samples, 1)
generated_y = diffusion.sample(batch_size=n_samples, conditioning=combined_conditioning)
y_coords = torch.cat([generated_y[0], generated_y[1]])


# create airfoil objects
airfoils = []
airfoil_coef = []
latent_vectors = []
max_camber_list = []
max_thickness_list = []
for i in range(n_samples):
    airfoil_y = generated_y[i]
    y_coords = torch.cat([airfoil_y[0], airfoil_y[1]])
    latent_vector = vae_model.enc(y_coords)
    latent_vector = torch.tensor(latent_vector[0])
    latent_vectors.append(latent_vector)
    y_coords = np.array(y_coords.detach().cpu())
    y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
    #y_coords = smooth_y_coords(y_coords, method='savitzky_golay', window_size=5, polyorder=3)
    coordinates = np.vstack([airfoil_x, y_coords]).T
    airfoil = asb.Airfoil(
        name = f'Generated Airfoil {i+1}',
        coordinates = coordinates
    )
    # normalize airfoils
    airfoil = airfoil.normalize()
    # repanel airfoils
    airfoil = airfoil.repanel(n_points_per_side=100)
    airfoils.append(airfoil)
    coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
    airfoil_coef.append(coef)
    max_camber = airfoil.max_camber()
    max_thickness = airfoil.max_thickness()
    max_camber_list.append(max_camber)
    max_thickness_list.append(max_thickness)

def plot_airfoils_grid(reconstructed_airfoils, airfoil_x, performance_coef, max_camber_list, max_thickness_list, grid_size=4):
    num_airfoils = reconstructed_airfoils.shape[0]
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    
    for i in range(num_airfoils):
        row = i // grid_size
        col = i % grid_size

        # Plot original airfoil
        ax = axs[row, col]
        reconstructed = reconstructed_airfoils[i].detach().cpu().numpy()
        normalized = airfoils[i].coordinates
        #ax.scatter(airfoil_x, reconstructed, color='black')
        ax.plot(normalized[:, 0], normalized[:, 1], color='black')
        ax.set_title(f'Generated Airfoil {i+1}, CL: {performance_coef[i]["CL"][0]:.4f}, CD: {performance_coef[i]["CD"][0]:.4f},\n Max Camber: {max_camber_list[i]:.4f},\n Max Thickness: {max_thickness_list[i]:.4f}')
        ax.set_aspect('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(airfoils) == 16:
    plot_airfoils_grid(generated_y, airfoil_x, airfoil_coef, max_camber_list, max_thickness_list, grid_size=4)

# Function to make histogram of cl and cd values
def plot_coef_hist():
    cl_values = [coef['CL'][0] for coef in airfoil_coef]
    cd_values = [coef['CD'][0] for coef in airfoil_coef]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()
    axs[0].hist(cl_values, bins=20, color='blue', alpha=0.5)
    axs[0].set_title('CL Values')
    axs[0].set_xlabel('CL')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(cd_values, bins=20, color='red', alpha=0.5)
    axs[1].set_title('CD Values')
    axs[1].set_xlabel('CD')
    axs[1].set_ylabel('Frequency')
    axs[2].hist(max_camber_list, bins=20, color='green', alpha=0.5)
    axs[2].set_title('Max Camber Values')
    axs[2].set_xlabel('Max Camber')
    axs[2].set_ylabel('Frequency')
    axs[3].hist(max_thickness_list, bins=20, color='purple', alpha=0.5)
    axs[3].set_title('Max Thickness Values')
    axs[3].set_xlabel('Max Thickness')
    axs[3].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

plot_coef_hist()

# get latent vectors of uiuc airfoils
uiuc_airfoils = []
for data in dataloader:
    airfoil = data['train_coords_y'].float().to(device)
    uiuc_airfoils.append(airfoil)
uiuc_latent_vectors = []

for airfoil in uiuc_airfoils:
    latent_vector = vae_model.enc(airfoil)
    latent_vector = latent_vector[0].detach().cpu().numpy()
    latent_vector = torch.tensor(latent_vector)
    uiuc_latent_vectors.append(latent_vector)

# make a bunch of random latent vectors
#random_latent_vectors = torch.randn(n_samples, latent_dim).to(device)
# pass random latent vectors through the VAE decoder
    

# plot t-sne of latent vectors
uiuc_latent_vectors = torch.stack(uiuc_latent_vectors)
uiuc_latent_vectors = uiuc_latent_vectors.squeeze(1)
tsne = TSNE(n_components=2)
latent_vectors = torch.stack(latent_vectors).detach().cpu()
latent_tsne = tsne.fit_transform(latent_vectors)
uiuc_tsne = tsne.fit_transform(uiuc_latent_vectors)
#random_tsne = tsne.fit_transform(random_latent_vectors.detach().cpu().numpy())
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], color='blue', label='Generated Airfoils', alpha=0.5)
plt.scatter(uiuc_tsne[:, 0], uiuc_tsne[:, 1], color='red', label='UIUC Airfoils', alpha=0.5)
#plt.scatter(random_tsne[:, 0], random_tsne[:, 1], color='green', label='Random Airfoils', alpha=0.5)
plt.title('t-SNE of Latent Vectors')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()