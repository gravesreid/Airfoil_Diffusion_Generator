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

train_vae_fresh = False

if train_vae_fresh:
    # Load vae training dataset
    vae_dataset = AirfoilDataset(path='aero_sandbox_processed/')
    vae_dataloader = DataLoader(vae_dataset, batch_size=64, shuffle=True)

    # train vae
    #beta was annealed from 0 to 0.001 over 1000 epochs to get better reconstructions
    #beta was annealed from 0.001 to 1.5 over 2000 epochs to get better latent space
    total_losses, vae, airfoil_x, mu = train_vae(device, vae_dataset, vae_dataloader, num_epochs=2000, learning_rate=0.006, beta_start=0.001, beta_end=1)

    # save vae model
    torch.save(vae.state_dict(), 'vae.pth')
    # Plot losses
    plot_losses(total_losses)

    # plot vae airfoils
    plot_all_airfoils(vae_dataset, vae, num_samples=16, latent_dim=32, device=device)

    # visualize latent space
    mu_tensor_pca, mu_tsne, labels, centroids_tsne = aggregate_and_cluster(vae, vae_dataloader, n_clusters=12, device=device, perplexity=50, n_iter=5000)


        # save vae recontstructions
    save_vae_reconstructions(vae_dataset, vae, device, output_dir='vae_reconstructed/')

    # reparameterize reconstructions
    #repanelize_airfoils('vae_reconstructed/', 'vae_recon_smoothed/', n_points_per_side=100)

    # evaluate reconstructed airfoils
    airfoil_path = "vae_reconstructed/"
    eval_path = 'vae_recon_eval/'
    neural_foil_eval(airfoil_path, eval_path)

    # calculate mean and standard deviation of cl and cd for vae reconstructions
    cl_mean, cl_std, cd_mean, cd_std = calculate_clcd_stats('vae_recon_eval/')

    # Plot Cl/Cd from original vs reconstructed airfoils
    plot_clcd_comparison('clcd_data/', 'vae_recon_eval/')
else:
    # Load the trained VAE model
    cl_mean, cl_std, cd_mean, cd_std = calculate_clcd_stats('vae_recon_eval/')
    airfoil_dim = 199
    latent_dim = 32
    in_channels = 1
    out_channels = 1
    vae = VAE(airfoil_dim, latent_dim)
    vae.load_state_dict(torch.load('vae.pth'))
    vae.to(device)
    vae.eval()


# train diffusion model on latent space
conditioned_dataset = ConditionedAirfoilDataset(path='aero_sandbox_processed/', eval_path='clcd_data/')
conditioned_dataloader = DataLoader(conditioned_dataset, batch_size=128, shuffle=True)

dataset = AirfoilDataset(path='aero_sandbox_processed/')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

latent_diffusion = False


if latent_diffusion:
    diffusion_model, diffusion_loss = train_diffusion_latent(conditioned_dataloader, vae, device,cl_mean, cl_std, cd_mean, cd_std, conditioning=False, lr=0.001, epochs = 10000, log_freq=100)
else:
    diffusion_model, diffusion_loss = train_diffusion(conditioned_dataloader, device,cl_mean, cl_std, cd_mean, cd_std, base_dim=64, timesteps=100, conditioning=False, lr=1e-3, epochs = 5000, log_freq=100) # 50000 epochs, 1e-3 lr was good. 75000 epochs, no change


# save diffusion model
torch.save(diffusion_model.state_dict(), 'diffusion.pth')
plot_losses(diffusion_loss)
