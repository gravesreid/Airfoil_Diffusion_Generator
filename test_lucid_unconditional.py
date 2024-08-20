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
from datetime import datetime

vae_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
diffusion_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_unconditional_run_1/best_model.pt"

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
n_samples = 2000
unet_dim = 12
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
conditioning_cl = torch.tensor([.25]).to(device).unsqueeze(1)
conditioning_cd = torch.tensor([.01]).to(device).unsqueeze(1)
conditioning_cm = torch.tensor([0.0]).to(device).unsqueeze(1)
conditioning_max_camber = torch.tensor([.1]).to(device).unsqueeze(1)
conditioning_max_thickness = torch.tensor([.2]).to(device).unsqueeze(1)
combined_conditioning = torch.cat([conditioning_cl, conditioning_cd, conditioning_cm, conditioning_max_camber, conditioning_max_thickness], dim=1)
print(f'conditioning shape: {conditioning_cl.shape}')

# Load the trained VAE model
vae_model = VAE(airfoil_dim, latent_dim).to(device)
vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
vae_model.eval()

# Load the trained diffusion model
diffusion_model = Unet1DConditional(unet_dim, cond_dim=5, channels=2, dim_mults=(1,2,4)).to(device)
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
conditioning_cl = conditioning_cl.repeat(n_samples, 1)
combined_conditioning = combined_conditioning.repeat(n_samples, 1)
generated_y = diffusion.sample(batch_size=n_samples, conditioning=None)
y_coords = torch.cat([generated_y[0], generated_y[1]])


# create airfoil objects
airfoils = []
airfoil_coef = []
gen_coords = []
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
    gen_coord = airfoil.coordinates
    gen_coords.append(gen_coord)
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


# get latent vectors of uiuc airfoils
uiuc_airfoils = []
uiuc_y_coords = []
uiuc_coords = []
uiuc_cl = []
uiuc_cd = []
uiuc_max_camber = []
uiuc_max_thickness = []
for data in dataloader:
    coordinates = data['coordinates'].float().to(device)
    coordinates = coordinates.squeeze(0).detach().cpu().numpy().T
    print(f'Coordinates shape: {coordinates.shape}')
    uiuc_coords.append(coordinates)
    airfoil_y = data['train_coords_y'].float().to(device)
    airfoil_y = airfoil_y.squeeze(0).detach().cpu().numpy()
    airfoil_y = np.concatenate([airfoil_y[0], airfoil_y[1]])
    print(f'Airfoil y shape: {airfoil_y.shape}')
    uiuc_y_coords.append(airfoil_y)
    uiuc_cl.append(data['CL'][0])
    uiuc_cd.append(data['CD'][0])
    uiuc_max_camber.append(data['max_camber'][0])
    uiuc_max_thickness.append(data['max_thickness'][0])
    airfoil = asb.Airfoil(
        name = 'UIUC Airfoil',
        coordinates = coordinates
    )
    uiuc_airfoils.append(airfoil.coordinates)
uiuc_latent_vectors = []

for airfoil in uiuc_y_coords:
    airfoil = torch.tensor(airfoil).to(device)
    latent_vector = vae_model.enc(airfoil)
    latent_vector = latent_vector[0].detach().cpu().numpy()
    latent_vector = torch.tensor(latent_vector)
    uiuc_latent_vectors.append(latent_vector)


# set fontsize for titles and labels
plt.rcParams.update({'font.size': 18})

def plot_coef_hist():
    cl_values = [coef['CL'][0] for coef in airfoil_coef]
    cd_values = [coef['CD'][0] for coef in airfoil_coef]

    # Assuming uiuc_cl, uiuc_cd, uiuc_max_camber, and uiuc_max_thickness are already defined
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()

    # Plot CL Values
    axs[0].hist(cl_values, bins=10, alpha=0.5, label='Generated Airfoils')
    axs[0].hist(uiuc_cl, bins=20, alpha=0.5, label='UIUC Airfoils')
    axs[0].set_title('CL Values', fontsize=24)
    axs[0].set_xticks(np.arange(-0.2, 1.75, 0.2))
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    for spine in axs[0].spines.values():
        spine.set_linewidth(2)
    axs[0].legend()

    # Plot CD Values
    axs[1].hist(cd_values, bins=5, alpha=0.5, label='Generated Airfoils')
    axs[1].hist(uiuc_cd, bins=20, alpha=0.5, label='UIUC Airfoils')
    axs[1].set_title('CD Values', fontsize=24)
    axs[1].set_xlim(0, 0.03)
    axs[1].set_xticks(np.arange(0.005, 0.04, 0.01))
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    for spine in axs[1].spines.values():
        spine.set_linewidth(2)
    axs[1].legend()

    # Plot Max Camber Values
    axs[2].hist(max_camber_list, bins=10, alpha=0.5, label='Generated Airfoils')
    axs[2].hist(uiuc_max_camber, bins=20, alpha=0.5, label='UIUC Airfoils')
    axs[2].set_title('Max Camber Values', fontsize=24)
    axs[2].set_xticks(np.arange(0, 0.15, 0.05))
    axs[2].tick_params(axis='both', which='major', labelsize=20)
    for spine in axs[2].spines.values():
        spine.set_linewidth(2)
    axs[2].legend()

    # Plot Max Thickness Values
    axs[3].hist(max_thickness_list, bins=10, alpha=0.5, label='Generated Airfoils')
    axs[3].hist(uiuc_max_thickness, bins=20, alpha=0.5, label='UIUC Airfoils')
    axs[3].set_title('Max Thickness Values', fontsize=24)
    axs[3].set_xticks(np.arange(0, 0.5, 0.1))
    axs[3].tick_params(axis='both', which='major', labelsize=20)
    for spine in axs[3].spines.values():
        spine.set_linewidth(2)
    axs[3].legend()

    plt.tight_layout()
    plt.show()
plot_coef_hist()

def plot_boxplots():
    cl_values = [coef['CL'][0] for coef in airfoil_coef]
    cd_values = [coef['CD'][0] for coef in airfoil_coef]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()

    # Plot CL Values
    axs[0].boxplot([cl_values, uiuc_cl], labels=['Generated Airfoils', 'UIUC Airfoils'])
    axs[0].set_title('CL Values')

    # Plot CD Values
    axs[1].boxplot([cd_values, uiuc_cd], labels=['Generated Airfoils', 'UIUC Airfoils'])
    axs[1].set_title('CD Values')

    # Plot Max Camber Values
    axs[2].boxplot([max_camber_list, uiuc_max_camber], labels=['Generated Airfoils', 'UIUC Airfoils'])
    axs[2].set_title('Max Camber Values')

    # Plot Max Thickness Values
    axs[3].boxplot([max_thickness_list, uiuc_max_thickness], labels=['Generated Airfoils', 'UIUC Airfoils'])
    axs[3].set_title('Max Thickness Values')

    plt.tight_layout()
    plt.show()

plot_boxplots()




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
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], color='blue', label='Generated Airfoils', alpha=0.5)
ax.scatter(uiuc_tsne[:, 0], uiuc_tsne[:, 1], color='red', label='UIUC Airfoils', alpha=0.5)
ax.set_title('t-SNE of UIUC and Generated Profile Latent Vectors', fontsize=22)
ax.set_xlabel('t-SNE Dimension 1', fontsize=20)
ax.set_ylabel('t-SNE Dimension 2', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.legend()
plt.tight_layout()
plt.show()

tsne_1_range = [40, 60]
tsne_2_range = [-5, 20]
selected_indices_generated = (latent_tsne[:, 0] > tsne_1_range[0]) & (latent_tsne[:, 0] < tsne_1_range[1]) & (latent_tsne[:, 1] > tsne_2_range[0]) & (latent_tsne[:, 1] < tsne_2_range[1])
print(f'Number of selected generated airfoils: {selected_indices_generated.sum()}')

# Get the corresponding generated airfoil profiles
selected_airfoils = [gen_coords[i] for i in range(len(gen_coords)) if selected_indices_generated[i]]
print(f'Number of selected generated airfoils: {len(selected_airfoils)}')

# Now, do the same for the UIUC airfoils using their t-SNE projection
selected_indices_uiuc = (uiuc_tsne[:, 0] > tsne_1_range[0]) & (uiuc_tsne[:, 0] < tsne_1_range[1]) & (uiuc_tsne[:, 1] > tsne_2_range[0]) & (uiuc_tsne[:, 1] < tsne_2_range[1])
print(f'Number of selected UIUC airfoils: {selected_indices_uiuc.sum()}')

# Get the corresponding UIUC airfoil profiles
selected_uiuc_airfoils = [uiuc_airfoils[i] for i in range(len(uiuc_airfoils)) if selected_indices_uiuc[i]]
print(f'Number of selected UIUC airfoils: {len(selected_uiuc_airfoils)}')

# Plot these selected airfoils in a grid
def plot_selected_airfoils(selected_airfoils, grid_size=4):
    num_airfoils = len(selected_airfoils)
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    axs = axs.flatten()
    
    for i in range(int(grid_size ** 2)):
        ax = axs[i]
        airfoil = selected_airfoils[i]
        ax.plot(airfoil[:, 0], airfoil[:, 1], color='black')
        ax.set_title(f'Selected Airfoil {i+1}')
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_selected_airfoils(selected_airfoils)
plot_selected_airfoils(selected_uiuc_airfoils)

# Save the plot to a file
def save_airfoil_plots(selected_airfoils, file_name):
    date = datetime.now().strftime("%Y-%m-%d")
    save_dir = f'/home/reid/Downloads/airfoil_plots_{date}'
    num_airfoils = len(selected_airfoils)
    grid_size = min(4, int(np.ceil(np.sqrt(num_airfoils))))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    axs = axs.flatten()
    
    for i in range(int(grid_size ** 2)):
        ax = axs[i]
        airfoil = selected_airfoils[i]
        ax.plot(airfoil[:, 0], airfoil[:, 1], color='black')
        ax.set_title(f'Selected Airfoil {i+1}')
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, file_name)
    plt.savefig(file_name)
    plt.close()

save_airfoil_plots(selected_airfoils, f'selected_generated_airfoils_{tsne_1_range[0]}_{tsne_1_range[1]}_{tsne_2_range[0]}_{tsne_2_range[1]}.png')
save_airfoil_plots(selected_uiuc_airfoils, f'selected_uiuc_airfoils_{tsne_1_range[0]}_{tsne_1_range[1]}_{tsne_2_range[0]}_{tsne_2_range[1]}.png')