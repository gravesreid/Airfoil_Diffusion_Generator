import matplotlib.pyplot as plt
from scipy.stats import linregress
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import subprocess
import os
import logging
import shutil
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import aerosandbox as asb
import neuralfoil as nf
import os

from models import *

def plot_losses(losses, title="Training Loss Over Time"):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_airfoils(airfoil_x, airfoil_y, title="Airfoil Plot"):
    idx = 0
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    for row in ax:
        for col in row:
            if idx >= len(airfoil_y):
                col.axis('off')
            else:
                y_plot = airfoil_y[idx].numpy() if isinstance(airfoil_y[idx], torch.Tensor) else airfoil_y[idx]
                col.scatter(airfoil_x, y_plot, s=0.6, c='black')
                col.axis('off')
                col.axis('equal')
                idx += 1
    plt.show()

def plot_airfoil(airfoil_x, airfoil_y, title="Airfoil Plot"):
    plt.figure(figsize=(10, 2))
    plt.scatter(airfoil_x, airfoil_y, s=0.6, c='black')
    plt.axis('off')
    plt.axis('equal')
    plt.title(title)
    plt.show()

def plot_all_airfoils(dataset, vae, num_samples=128, latent_dim=32, device='cpu'):
    """
    Plots real, reconstructed, and synthesized airfoils using the provided dataset and VAE model.
    
    Parameters:
        dataset (Dataset): The dataset object containing the airfoils.
        vae (nn.Module): The VAE model.
        num_samples (int): Number of samples to plot.
        latent_dim (int): Latent dimension of the VAE model.
        device (str): Device to perform computations on.
    """
    airfoil_x = dataset.get_x()  # Assuming this returns the common x-coordinates
    real_airfoils = dataset.get_y()[:num_samples]
    
    # Convert to tensor and pad if necessary
    if all(y.shape == real_airfoils[0].shape for y in real_airfoils):
        real_airfoils_tensor = torch.stack(real_airfoils).to(device)
    else:
        max_length = max(y.size(0) for y in real_airfoils)
        real_airfoils_tensor = torch.stack([nn.functional.pad(y, (0, max_length - y.size(0))) for y in real_airfoils]).to(device)
    
    # Generate reconstructed airfoils
    recon_airfoils, _, _ = vae(real_airfoils_tensor)
    recon_airfoils = recon_airfoils.detach().cpu().numpy()
    
    # Generate synthesized airfoils
    noise = torch.randn(num_samples, latent_dim).to(device)
    gen_airfoils = vae.decode(noise).detach().cpu().numpy()
    
    # Plot real, reconstructed, and synthesized airfoils
    plot_airfoils(airfoil_x, real_airfoils, title="Real Airfoils")
    plot_airfoils(airfoil_x, recon_airfoils, title="Reconstructed Airfoils")
    plot_airfoils(airfoil_x, gen_airfoils, title="Synthesized Airfoils")


def visualize_latent_space(vae, data_loader, n_iter = 5000, perplexity = 100, device = 'cpu'):
    mu_list = []
    for _,data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            mu, _ = vae.enc(data)
            mu_list.append(mu)

    mu_tensor = torch.cat(mu_list, dim=0)
    mu_tensor = mu_tensor.view(mu_tensor.size(0), -1)

    tsne = TSNE(n_components=2, random_state=42, n_iter=n_iter, perplexity=perplexity)
    mu_tsne = tsne.fit_transform(mu_tensor.cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], alpha=0.5)
    plt.title('Latent Space Visualization using t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()

    return mu_tensor, mu_tsne

def find_interpolated_vector(t_sne_output, latent_vectors, point, k=5, device='cpu'):
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.cpu().numpy()
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(t_sne_output)
    distances, indices = nbrs.kneighbors([point])
    
    latent_points = latent_vectors[indices[0]]
    weights = 1 / distances[0]
    interpolated_vector = np.average(latent_points, axis=0, weights=weights)
    interpolated_vector_tensor = torch.tensor(interpolated_vector, dtype=torch.float32).to(device)
    
    return interpolated_vector_tensor

def plot_latent_space_airfoils(vae, airfoil_x, mu_tensor, t_sne_output, device, grid_points):
    """
    Plots a grid of airfoils corresponding to specified points in the latent space t-SNE output.

    Parameters:
        vae (nn.Module): Trained VAE model.
        airfoil_x (Tensor): X coordinates for airfoil plotting.
        mu_tensor (Tensor): Tensor of latent vectors from the VAE encoder.
        t_sne_output (np.array): t-SNE reduced latent vectors.
        device (str): Computation device ('cpu' or 'cuda').
        grid_points (list of lists): Points in the latent space to plot ([x, y] pairs).
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))  # Set up a 3x3 grid
    fig.suptitle('Airfoils from Selected Latent Space Points', fontsize=16)

    # Iterate over the grid and plot airfoils
    for ax, point in zip(axes.flat, grid_points):
        interpolated_vector = find_interpolated_vector(t_sne_output, mu_tensor, point, device=device)
        decoder_output = vae.decode(interpolated_vector).detach().cpu().numpy()

        # Ensure correct format for plotting
        if isinstance(decoder_output, torch.Tensor):
            decoder_output = decoder_output.numpy()

        ax.scatter(airfoil_x, decoder_output, s=0.6, c='black')
        ax.axis('off')
        ax.axis('equal')
        ax.set_title(f"Point: {point}")

    plt.show()

def save_vae_reconstructions(dataset, vae, device, output_dir='vae_reconstructed'):
    """
    Processes each sample in the dataset using a VAE, reconstructs the airfoils, 
    and saves them in a .dat format.

    Parameters:
        dataset (Dataset): The dataset to process.
        vae (nn.Module): The trained VAE model.
        device (str): Device to perform computations ('cpu' or 'cuda').
        output_dir (str): Directory to save reconstructed .dat files.
    """
    # Ensure the model is in evaluation mode
    vae.eval()
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create directory for outputs if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get x_positions for formatting output files
    x_positions = dataset.get_x()

    with torch.no_grad():
        # Process each sample and save reconstruction
        for batch_idx, (name, local_batch) in enumerate(data_loader):
            y_real = local_batch.to(device)
            recon_y, _, _ = vae(y_real)
            y_values = recon_y.squeeze().cpu().numpy()

            # Combine x and y positions and transpose for saving
            combined_data = np.vstack((x_positions, y_values)).T
            
            # Format and save reconstructed output in .dat format
            save_path = os.path.join(output_dir, f'{name[0]}.dat')
            with open(save_path, 'w') as f:
                np.savetxt(f, combined_data, fmt='%1.7e', delimiter='   ')
            
            # Optionally print progress
            print(f"Saved reconstruction {batch_idx + 1}/{len(data_loader)}")


def neural_foil_eval(path, clcd_path, alpha=0, Re=2e6, mach=0.2):
    """
    Evaluates the aerodynamic coefficients Cl and Cd for each airfoil in the specified directory.
    
    Parameters:
        path (str): Directory containing the .dat files.
        clcd_path (str): Directory to save the Cl/Cd data.
        alpha (float): Angle of attack in degrees.
        Re (float): Reynolds number.
        mach (float): Mach number.
    """
    # List all .dat files in the specified directory
    files = [f for f in os.listdir(path) if f.endswith('.dat')]

    files_processed = 0
    max_cl = 0
    max_cl_name = ''
    max_cd = 0
    max_cd_name = ''
    
    # Iterate over each file
    for filename in files:
        try:
            # Calculate the aerodynamic coefficients
            print(filename)
            coef = nf.get_aero_from_dat_file(
                os.path.join(path, filename),
                alpha=alpha,
                Re=Re,
                model_size="xxxlarge",
            )
            cl = coef['CL'][0]
            cd = coef['CD'][0]
            if cl > max_cl:
                max_cl = cl
                max_cl_name = filename
            if cd > max_cd:
                max_cd = cd
                max_cd_name = filename
            combined_data = np.vstack((cl, cd)).T
            print(f"CL: {cl:.4f}, CD: {cd:.4f}")
            np.savetxt(os.path.join(clcd_path, os.path.splitext(filename)[0] + '.csv'), combined_data, delimiter=',')
            
            # Print the airfoil object and aerodynamic coefficients
            print(filename)
            print(f'CL: {cl:.4f}, CD: {cd:.4f}')
            files_processed += 1
        except Exception as e:
            print(f'Error processing {filename}: {e}')
            continue
    print(f'Processed {files_processed} files')
    print(f'Max CL: {max_cl} for {max_cl_name}')
    print(f'Max CD: {max_cd} for {max_cd_name}')


def plot_clcd_comparison(clcd_path_1, clcd_path_2):
    """
    Plots a comparison of Cl/Cd data from two directories with percent error.
    
    Parameters:
        clcd_path_1 (str): Directory containing the first set of Cl/Cd data.
        clcd_path_2 (str): Directory containing the second set of Cl/Cd data.
    """
    # List all .csv files in the specified directories
    files_1 = {f: os.path.join(clcd_path_1, f) for f in os.listdir(clcd_path_1) if f.endswith('.csv')}
    files_2 = {f: os.path.join(clcd_path_2, f) for f in os.listdir(clcd_path_2) if f.endswith('.csv')}
    
    common_files = set(files_1.keys()).intersection(files_2.keys())  # Ensure only comparing files that exist in both
    #print(f'Common files: {common_files}')
    CL_err = []
    CD_err = []

    max_cl_err = 0
    max_cl_err_name = ''
    max_cd_err = 0
    max_cd_err_name = ''

    for filename in common_files:
        data_1 = np.loadtxt(files_1[filename], delimiter=',')
        data_2 = np.loadtxt(files_2[filename], delimiter=',')
        
        if data_1[0] != 0:  # Avoid division by zero
            cl_error = abs((data_1[0] - data_2[0]) / data_1[0]) * 100
            CL_err.append(cl_error)
            if cl_error > max_cl_err:
                max_cl_err = cl_error
                max_cl_err_name = filename
        if data_1[1] != 0:  # Avoid division by zero
            cd_error = abs((data_1[1] - data_2[1]) / data_1[1]) * 100
            CD_err.append(cd_error)
            if cd_error > max_cd_err:
                max_cd_err = cd_error
                max_cd_err_name = filename
    print(f'Max CL Error: {max_cl_err} for {max_cl_err_name}')
    print(f'Max CD Error: {max_cd_err} for {max_cd_err_name}')

    indices = np.arange(len(CL_err))

    # Plot the Cl/Cd percent error
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(common_files, CL_err)
    plt.xlabel('Index')
    plt.ylabel('Percent Error (%)')
    plt.title('CL Percent Error')
    
    plt.subplot(1, 2, 2)
    plt.scatter(common_files, CD_err)
    plt.xlabel('Index')
    plt.ylabel('Percent Error (%)')
    plt.title('CD Percent Error')

    plt.show()

def train_vae(device, dataset, dataloader, num_epochs=200, learning_rate=0.001, beta_start=0.0, beta_end=0.01):

    # Get dataset dimensions
    airfoil_x = dataset.get_x()  # Used for plotting later
    airfoil_dim = airfoil_x.shape[0]
    latent_dim = 32

    # Initialize the model
    vae = VAE(airfoil_dim, latent_dim).to(device)

    # Define loss function
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + beta * KLD

    # Optimizer and Scheduler
    optim = Adam(vae.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.75, patience=20, verbose=True)

    # Beta scheduling
    beta_increment = (beta_end - beta_start) / num_epochs

    # Training loop
    total_losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        current_beta = beta_start + beta_increment * epoch
        for _,local_batch in dataloader:
            y_real = local_batch.to(device)
            recon_y, mu, logvar = vae(y_real)
            loss = loss_function(recon_y, y_real, mu, logvar, beta=current_beta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        scheduler.step(sum(epoch_losses) / len(epoch_losses))
        total_losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f'Epoch {epoch+1}: Total Loss: {total_losses[-1]:.4f}, Learning Rate: {optim.param_groups[0]["lr"]:.6f}')

    return total_losses, vae, airfoil_x

def train_diffusion(conditioned_dataloader, device, lr=0.01, epochs=500, log_freq=50):
    """
    Trains a diffusion model on conditioned airfoil data.

    Parameters:
        path (str): Directory containing training data.
        eval_path (str): Directory containing evaluation data.
        device (torch.device): The device (GPU/CPU) to run the training on.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loading.
        log_freq (int): Frequency of logging training progress.
    """

    # Initialize the model and move it to the specified device
    model = AirfoilDiffusion(32, 1, 1).to(device)

    # Set up the optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10)
    loss_fn = nn.L1Loss()

    # Training loop
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        for _,data, cl_cd in conditioned_dataloader:
            data = data.to(device)
            cl_cd = cl_cd.to(device)
            pred = model(data)
            loss = loss_fn(pred, cl_cd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
        
        epoch_loss /= len(conditioned_dataloader)
        scheduler.step(epoch_loss)

        if i % log_freq == 0 or i == epochs - 1:
            print(f"Epoch {i} loss: {epoch_loss:.4f}, learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    return model



def repanelize_airfoils(input_dir, output_dir, n_points_per_side=100):
    """
    Processes airfoil data files by repaneling and saving to a new directory.
    
    Parameters:
        input_dir (str): Directory containing the original .dat files.
        output_dir (str): Directory where the processed .dat files will be saved.
        n_points_per_side (int): Number of points per side for repaneling the airfoil.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all .dat files in the specified directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
    
    # Iterate over each file
    for filename in files:
        try:
            # Load the airfoil from a .dat file
            af = asb.Airfoil(name=filename[:-4], file=os.path.join(input_dir, filename))
            print(f'Loaded {af}')
            
            # Repanel the airfoil
            af = af.repanel(n_points_per_side=n_points_per_side)
            
            # Write the repaneled airfoil data back to a new .dat file
            af.write_dat(filepath=os.path.join(output_dir, filename))
            
            # Print the updated airfoil object
            print(f'Processed and saved {af}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')

    