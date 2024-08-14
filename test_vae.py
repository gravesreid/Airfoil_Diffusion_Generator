import torch
import os
from torch.utils.data import DataLoader
from airfoil_dataset_1d import AirfoilDataset
import matplotlib.pyplot as plt
from utils_1d import *
from vae import *
import aerosandbox as asb

airfoil_dim = 200
latent_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = VAE(airfoil_dim, latent_dim).to(device)
model_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

# Initialize dataset and dataloader
airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

airfoil_x = dataset.get_x()

def plot_airfoils_grid(original_airfoils, reconstructed_airfoils, airfoil_x, grid_size=4):
    num_airfoils = original_airfoils.shape[0]
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    
    for i in range(num_airfoils):
        row = i // grid_size
        col = i % grid_size

        # Plot original airfoil
        ax = axs[row, col]
        original = original_airfoils[i].detach().cpu().numpy()
        ax.scatter(airfoil_x, original, color='black') 
        reconstructed = reconstructed_airfoils[i].detach().cpu().numpy()
        ax.scatter(airfoil_x, reconstructed, color='red')
        ax.set_title(f'Original Airfoil {i+1}')
        ax.set_aspect('equal')
        ax.axis('off')

        # Plot reconstructed airfoil
       # ax = axs[row * 2 + 1, col]
       # reconstructed = reconstructed_airfoils[i].detach().cpu().numpy()
       # ax.scatter(airfoil_x, reconstructed, color='red')
       # ax.set_title(f'Reconstructed Airfoil {i+1}')
       # ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()



# plot a 4x4 grid of all the airfoils
original_airfoils = []
reconstructed_airfoils = []
for i, data in enumerate(dataloader):
    airfoil = data['train_coords_y'].float().to(device)
    reconstructed_airfoil, _, _ = model(airfoil)
    reconstructed_airfoil = reconstructed_airfoil.detach().cpu().numpy()
    airfoil = airfoil.detach().cpu().numpy()
    original_airfoils.append(airfoil)
    reconstructed_airfoils.append(reconstructed_airfoil)


def visualize_vae_reconstructiotns(model, dataloader, airfoil_x):  
    for data in dataloader:
        airfoil = data['train_coords_y'].float().to(device)
        reconstructed_airfoil, _, _ = model(airfoil)
#
        plot_airfoils_grid(airfoil, reconstructed_airfoil, airfoil_x)

def get_original_performance_coef(model, dataset):
    cl_list = []
    cd_list = []
    for data in dataset:
        cl = data['CL']
        cd = data['CD']
        cl_list.append(cl)
        cd_list.append(cd)
    return cl_list, cd_list


visualize_vae_reconstructiotns(model, dataloader, airfoil_x)

# test new generations
# Generate new airfoils
num_airfoils = 1600
def generate_airfoils(model, num_airfoils, latent_dim, device):
    z = torch.randn(num_airfoils, latent_dim).to(device)
    generated_airfoils = model.decode(z)
    # create airfoil objects from the generated airfoils
    airfoils = []
    cl_list = []
    cd_list = []
    for i in range(num_airfoils):
        airfoil_y = generated_airfoils[i].detach().cpu().numpy()
        coordinates = np.vstack((airfoil_x, airfoil_y)).T
        airfoil = asb.Airfoil(coordinates=coordinates)
        # get performance metrics
        coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=2e6, mach=0.2)
        cl_list.append(coef['CL'][0])
        cd_list.append(coef['CD'][0])
        airfoils.append(airfoil)
    original_cl, original_cd = get_original_performance_coef(model, dataset)


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cl_list, bins=20, alpha=0.5, label='Generated CL')
    ax.hist(original_cl, bins=20, alpha=0.5, label='Original CL')
    ax.legend()
    ax.set_title('CL Distribution')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cd_list, bins=20, alpha=0.5, label='Generated CD')
    ax.hist(original_cd, bins=20, alpha=0.5, label='Original CD')
    ax.legend()
    ax.set_xlim(0, 0.01)
    ax.set_title('CD Distribution')
    plt.show()

generate_airfoils(model, num_airfoils, latent_dim, device)