import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import AirfoilDataset
from conditionedDataset import ConditionedAirfoilDataset
from utils import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load vae training dataset
vae_dataset = AirfoilDataset(path='aero_sandbox_processed/')
vae_dataloader = DataLoader(vae_dataset, batch_size=16, shuffle=True)

# train vae
total_losses, vae, airfoil_x = train_vae(device, vae_dataset, vae_dataloader, num_epochs=200, learning_rate=0.001, beta_start=0.0, beta_end=0.01)

# Plot losses
plot_losses(total_losses)

# plot vae airfoils
plot_all_airfoils(vae_dataset, vae, num_samples=16, latent_dim=32, device=device)

# visualize latent space
grid_points = [[-4, -4], [-4, 0], [-4, 4], [0, -4], [0, 0], [0, 4], [4, -4], [4, 0], [4, 4]]
mu_tensor, t_sne_output = visualize_latent_space(vae, vae_dataloader, n_iter=5000, perplexity = 500, device=device)
plot_latent_space_airfoils(vae, vae_dataset.get_x(), mu_tensor, t_sne_output, device, grid_points)

# save vae recontstructions
save_vae_reconstructions(vae_dataset, vae, device, output_dir='vae_reconstructed/')

# reparameterize reconstructions
#repanelize_airfoils('vae_reconstructed/', 'vae_recon_smoothed/', n_points_per_side=100)

# evaluate reconstructed airfoils
airfoil_path = "vae_reconstructed/"
eval_path = 'vae_recon_eval/'
neural_foil_eval(airfoil_path, eval_path)

# Plot Cl/Cd from original vs reconstructed airfoils
plot_clcd_comparison('clcd_data/', 'vae_recon_eval/')


# train diffusion model on latent space
conditioned_dataset = ConditionedAirfoilDataset(path='vae_reconstructed/', eval_path='vae_recon_eval/')
conditioned_dataloader = DataLoader(conditioned_dataset, batch_size=16, shuffle=True)

diffusion_model, diffusion_loss = train_diffusion(conditioned_dataloader, vae, device, lr=0.01, epochs = 500, log_freq=10)
plot_losses(diffusion_loss)
