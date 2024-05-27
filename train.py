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
vae_dataloader = DataLoader(vae_dataset, batch_size=128, shuffle=True)

# train vae
#beta was annealed from 0 to 0.001 over 1000 epochs to get better reconstructions
total_losses, vae, airfoil_x = train_vae(device, vae_dataset, vae_dataloader, num_epochs=2000, learning_rate=0.01, beta_start=0.001, beta_end=.5)

# save vae model
torch.save(vae.state_dict(), 'vae.pth')
# Plot losses
plot_losses(total_losses)

# plot vae airfoils
plot_all_airfoils(vae_dataset, vae, num_samples=16, latent_dim=32, device=device)

# visualize latent space
mu_pca, mu_tsne, labels, centroids_tsne = aggregate_and_cluster(vae, vae_dataloader, n_clusters=9, device=device, perplexity=30, n_iter=5000)
plot_latent_space_airfoils(vae, vae_dataset, centroids_tsne, device=device)

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
# save diffusion model
torch.save(diffusion_model.state_dict(), 'diffusion.pth')
plot_losses(diffusion_loss)
