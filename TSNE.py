import torch
import matplotlib.pyplot as plt
from dataset import AirfoilDataset
from utils import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define airfoil dimensions and latent size (ensure these match your training setup)
airfoil_dim = 199
latent_dim = 32
in_channels = 1
out_channels = 1

# Load the trained VAE model
vae = VAE(airfoil_dim, latent_dim)
vae.load_state_dict(torch.load('vae.pth'))
vae.to(device)
vae.eval()


vae_dataset = AirfoilDataset(path='aero_sandbox_processed/')
vae_dataloader = DataLoader(vae_dataset, batch_size=64, shuffle=True)
mu_tensor_pca, mu_tsne, labels, centroids_tsne = aggregate_and_cluster(vae, vae_dataloader, n_clusters=12, device=device, perplexity=50, n_iter=500)