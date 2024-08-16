import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from airfoil_dataset_1d_2channel import AirfoilDataset
from torch.utils.data import DataLoader
from vae import VAE
from LucidDiffusion import *
import aerosandbox as asb
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


vae_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
diffusion_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_run_10/best_model.pt"

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

# Initialize dataset
airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load UIUC airfoils and extract statistics
uiuc_airfoils = []
cl_list = []
cd_list = []
cm_list = []
max_camber_list = []
max_thickness_list = []
for data in dataloader:
    airfoil = data['train_coords_y'].float().to(device)
    cl = data['CL'].float().to(device)
    cd = data['CD'].float().to(device)
    cm = data['CM'].float().to(device)
    max_camber = data['max_camber'].float().to(device)
    max_thickness = data['max_thickness'].float().to(device)
    uiuc_airfoils.append(airfoil)
    cl_list.append(cl)
    cd_list.append(cd)
    cm_list.append(cm)
    max_camber_list.append(max_camber)
    max_thickness_list.append(max_thickness)

# Calculate mean, std of cl, cd, cm, max_camber, max_thickness of uiuc airfoils
uiuc_mean_cl = torch.mean(torch.stack(cl_list))
uiuc_mean_cd = torch.mean(torch.stack(cd_list))
uiuc_mean_cm = torch.mean(torch.stack(cm_list))
uiuc_mean_max_camber = torch.mean(torch.stack(max_camber_list))
uiuc_mean_max_thickness = torch.mean(torch.stack(max_thickness_list))

uiuc_std_cl = torch.std(torch.stack(cl_list))
uiuc_std_cd = torch.std(torch.stack(cd_list))
uiuc_std_cm = torch.std(torch.stack(cm_list))
uiuc_std_max_camber = torch.std(torch.stack(max_camber_list))
uiuc_std_max_thickness = torch.std(torch.stack(max_thickness_list))

# Calculate recommended ranges
cl_range = (uiuc_mean_cl - 2 * uiuc_std_cl, uiuc_mean_cl + 2 * uiuc_std_cl)
cd_range = (uiuc_mean_cd - 2 * uiuc_std_cd, uiuc_mean_cd + 2 * uiuc_std_cd)
cm_range = (uiuc_mean_cm - 2 * uiuc_std_cm, uiuc_mean_cm + 2 * uiuc_std_cm)
max_camber_range = (uiuc_mean_max_camber - 2 * uiuc_std_max_camber, uiuc_mean_max_camber + 2 * uiuc_std_max_camber)
max_thickness_range = (uiuc_mean_max_thickness - 2 * uiuc_std_max_thickness, uiuc_mean_max_thickness + 2 * uiuc_std_max_thickness)

# Display recommended ranges to the user
print(f"Recommended CL range: {cl_range[0]:.4f} to {cl_range[1]:.4f}")
print(f"Recommended CD range: {cd_range[0]:.4f} to {cd_range[1]:.4f}")
print(f"Recommended CM range: {cm_range[0]:.4f} to {cm_range[1]:.4f}")
print(f"Recommended Max Camber range: {max_camber_range[0]:.4f} to {max_camber_range[1]:.4f}")
print(f"Recommended Max Thickness range: {max_thickness_range[0]:.4f} to {max_thickness_range[1]:.4f}")

def get_user_conditioning():
    cl = float(input(f"Enter desired CL value within range ({cl_range[0]:.4f} to {cl_range[1]:.4f}): "))
    cd = float(input(f"Enter desired CD value within range ({cd_range[0]:.4f} to {cd_range[1]:.4f}): "))
    cm = float(input(f"Enter desired CM value within range ({cm_range[0]:.4f} to {cm_range[1]:.4f}): "))
    max_camber = float(input(f"Enter desired Max Camber value within range ({max_camber_range[0]:.4f} to {max_camber_range[1]:.4f}): "))
    max_thickness = float(input(f"Enter desired Max Thickness value within range ({max_thickness_range[0]:.4f} to {max_thickness_range[1]:.4f}): "))
    return torch.tensor([cl, cd, cm, max_camber, max_thickness]).to(device).unsqueeze(0)

# Get user input
combined_conditioning = get_user_conditioning()
combined_conditioning = combined_conditioning.repeat(n_samples, 1)

# Load the trained VAE model
vae_model = VAE(airfoil_dim, latent_dim).to(device)
vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
vae_model.eval()

# Load the trained diffusion model
diffusion_model = Unet1DConditional(unet_dim, cond_dim=5, channels=2, dim_mults=(1,2,4)).to(device)
diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
diffusion_model.eval()
diffusion = GaussianDiffusion1D(diffusion_model, seq_length=latent_dim).to(device)

airfoil_x = dataset.get_x()

# Generate 16 samples
generated_y = diffusion.sample(batch_size=n_samples, conditioning=combined_conditioning)

# Process each generated sample
airfoils = []
airfoil_coef = []
for i in range(n_samples):
    airfoil_y = generated_y[i]
    y_coords = torch.cat([airfoil_y[0], airfoil_y[1]])
    y_coords = np.array(y_coords.detach().cpu())
    y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
    
    coordinates = np.vstack([airfoil_x, y_coords]).T
    airfoil = asb.Airfoil(name=f'Generated Airfoil {i+1}', coordinates=coordinates)
    airfoil = airfoil.normalize().repanel(n_points_per_side=100)
    airfoils.append(airfoil)
    
    coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
    airfoil_coef.append(coef)

# Select the 4 samples closest to the user-provided conditioning values
def calculate_error(coef, conditioning):
    cl_error = abs(coef['CL'][0] - conditioning[0].item())
    cd_error = abs(coef['CD'][0] - conditioning[1].item())
    cm_error = abs(coef['CM'][0] - conditioning[2].item())
    return cl_error + cd_error + cm_error 

errors = [calculate_error(coef, combined_conditioning[0]) for coef in airfoil_coef]
sorted_indices = np.argsort(errors)[:4]

selected_airfoils = [airfoils[i] for i in sorted_indices]
selected_coef = [airfoil_coef[i] for i in sorted_indices]

# Find the closest UIUC airfoil
def find_closest_uiuc_airfoil(cl, cd):
    min_error = float('inf')
    closest_airfoil = None
    closest_coef = None
    for i, coef in enumerate(airfoil_coef):
        error = abs(coef['CL'][0] - cl) + abs(coef['CD'][0] - cd)
        if error < min_error:
            min_error = error
            closest_airfoil = uiuc_airfoils[i]
            closest_coef = coef
    return closest_airfoil, closest_coef

desired_cl = combined_conditioning[0, 0].item()
desired_cd = combined_conditioning[0, 1].item()
closest_airfoil, closest_coef = find_closest_uiuc_airfoil(desired_cl, desired_cd)

# Plot the selected airfoils along with the closest UIUC airfoil
def plot_airfoils_with_uiuc(selected_airfoils, selected_coef, closest_airfoil, closest_coef):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    
    for i, ax in enumerate(axs[:3]):
        airfoil = selected_airfoils[i]
        coef = selected_coef[i]
        normalized = airfoil.coordinates
        
        ax.plot(normalized[:, 0], normalized[:, 1], color='black')
        ax.set_title(f'Airfoil {sorted_indices[i]+1}, CL: {coef["CL"][0]:.4f}, CD: {coef["CD"][0]:.4f}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Plot the closest UIUC airfoil
    airfoil_y = closest_airfoil.squeeze(0)
    airfoil_y = airfoil_y.detach().cpu().numpy()
    airfoil_y = np.concatenate([airfoil_y[0], airfoil_y[1]])
    coordinates = np.vstack([airfoil_x, airfoil_y]).T
    
    axs[3].plot(coordinates[:, 0], coordinates[:, 1], color='blue')
    axs[3].set_title(f'Closest UIUC Airfoil, CL: {closest_coef["CL"][0]:.4f}, CD: {closest_coef["CD"][0]:.4f}')
    axs[3].set_aspect('equal')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

plot_airfoils_with_uiuc(selected_airfoils, selected_coef, closest_airfoil, closest_coef)


