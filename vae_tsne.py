import torch
import os
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from airfoil_dataset_1d import AirfoilDataset
from vae import VAE
import numpy as np

# Parameters
airfoil_dim = 200
latent_dim = 100

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained VAE model
model = VAE(airfoil_dim, latent_dim).to(device)
model_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize dataset and dataloader
airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Extract latent vectors and cl/cd values from the entire dataset
latent_vectors = []
cl_values = []
cd_values = []

for data in dataloader:
    airfoil = data['train_coords_y'].float().to(device)
    cl = data['CL'].float().to(device)
    cd = data['CD'].float().to(device)
    mu, _ = model.enc(airfoil)  # Get the mean (mu) of the latent distribution
    latent_vectors.append(mu.detach().cpu().numpy())
    cl_values.append(cl.detach().cpu().numpy())
    cd_values.append(cd.detach().cpu().numpy())

# Convert lists to NumPy arrays
latent_vectors = np.concatenate(latent_vectors, axis=0)
cl_values = np.concatenate(cl_values, axis=0)
cd_values = np.concatenate(cd_values, axis=0)

# Apply t-SNE to reduce the latent space to 2D
tsne = TSNE(n_components=2, perplexity=30, n_iter=5000, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

# Define vmin and vmax to limit the color range for CL and CD values
cl_vmin = np.percentile(cl_values, 5)  # 5th percentile
cl_vmax = np.percentile(cl_values, 95)  # 95th percentile
cd_vmin = np.percentile(cd_values, 5)  # 5th percentile
cd_vmax = np.percentile(cd_values, 95)  # 95th percentile

# Plot the 2D t-SNE results colored by CL values with limited color range
plt.figure(figsize=(10, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=15, c=cl_values, cmap='viridis', vmin=cl_vmin, vmax=cl_vmax)
plt.colorbar(label='CL Value')
plt.title('t-SNE Visualization of VAE Latent Space Colored by CL (Limited Range)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# Optionally, plot the 2D t-SNE results colored by CD values with limited color range
plt.figure(figsize=(10, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=15, c=cd_values, cmap='plasma', vmin=cd_vmin, vmax=cd_vmax)
plt.colorbar(label='CD Value')
plt.title('t-SNE Visualization of VAE Latent Space Colored by CD (Limited Range)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()


