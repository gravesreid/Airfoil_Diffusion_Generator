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
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from MakeDatasets import generate_airfoils_cd, generate_airfoils_cl, generate_airfoils_cl_cd, generate_airfoils_all, generate_airfoils_cl_thickness
from tqdm import tqdm
from utils_1d_2channel import *

# load uiuc airfoils
uiuc_dict = load_uiuc_airfoils()
print(f'uiuc dict keys: {uiuc_dict.keys()}')
# print max and min for cl, max thickness and max camber
print(f"max cl: {uiuc_dict['uiuc_max_cl']}, min cl: {uiuc_dict['uiuc_min_cl']}")
print(f"max thickness: {uiuc_dict['uiuc_max_thickness']}, max camber: {uiuc_dict['uiuc_max_camber']}")
print(f'min thickness: {uiuc_dict["uiuc_min_thickness"]}, min camber: {uiuc_dict["uiuc_min_camber"]}')

thickness_list = np.linspace(0.02, 0.4, 8, dtype=np.float32)
cl_list = [-0.2, 0.0, .5, 1.0, 1.5, 2.0, 2.5, 3.0]
camber_list = np.ones(8, dtype=np.float32) * 0.3

airfoil_path = 'coord_seligFmt'
dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
airfoil_x = dataset.get_x()

# load the models
all_model_path = 'models/lucid_all_standardized_run_1/best_model.pt'
thickness_path = 'models/lucid_thickness_standardized_run_1/best_model.pt'
cl_path = 'models/lucid_cl_standardized_run_1/best_model.pt'
camber_path = 'models/lucid_camber_standardized_run_1/best_model.pt'
thickness_camber_path = 'models/lucid_thickness_camber_standardized_run_1/best_model.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_model = Unet1DConditional(12, cond_dim=3, channels=2, dim_mults=(1,2,4)).to(device)
all_model.load_state_dict(torch.load(all_model_path, weights_only=True))
all_model.eval()
all_model_diffusion = GaussianDiffusion1D(all_model, seq_length=100).to(device)

thickness_model = Unet1DConditional(16, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
thickness_model.load_state_dict(torch.load(thickness_path, weights_only=True))
thickness_model.eval()
thickness_model_diffusion = GaussianDiffusion1D(thickness_model, seq_length=100).to(device)

cl_model = Unet1DConditional(12, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
cl_model.load_state_dict(torch.load(cl_path, weights_only=True))
cl_model.eval()
cl_model_diffusion = GaussianDiffusion1D(cl_model, seq_length=100).to(device)

camber_model = Unet1DConditional(12, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
camber_model.load_state_dict(torch.load(camber_path, weights_only=True))
camber_model.eval()
camber_model_diffusion = GaussianDiffusion1D(camber_model, seq_length=100).to(device)

thickness_camber_model = Unet1DConditional(16, cond_dim=2, channels=2, dim_mults=(1,2,4)).to(device)
thickness_camber_model.load_state_dict(torch.load(thickness_camber_path, weights_only=True))
thickness_camber_model.eval()
thickness_camber_model_diffusion = GaussianDiffusion1D(thickness_camber_model, seq_length=100).to(device)


model_type = 'thickness_camber'
plot = True
pca = False
tsne = False
if model_type == 'thickness':
    thickness_batch = torch.tensor(thickness_list).to(device)
    thickness_batch = standardize_conditioning_values(thickness_batch, uiuc_dict['uiuc_max_thickness_mean'], uiuc_dict['uiuc_max_thickness_std'])
    thickness_batch = thickness_batch + 2
    airfoils = thickness_model_diffusion.sample(batch_size=len(thickness_list), conditioning=thickness_batch.unsqueeze(1).to(device))
    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[i, 1].cpu().detach().numpy()
            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, airfoil_x, s=0.1)
            ax.plot(airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f'thickness_{thickness_list[i]}')
        plt.show()
    if pca:
        pca_1 = PCA(n_components=2)
        airfoil_matrix = airfoils.view(len(thickness_list), -1).cpu().detach().numpy()
        transformed_data = pca_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        ax.set_title('PCA')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()
    if tsne:
        tsne_1 = TSNE(n_components=2)
        transformed_data = tsne_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        ax.set_title('TSNE')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()

elif model_type == 'thickness_camber':
    thickness_batch = torch.tensor(thickness_list).to(device)
    thickness_batch = standardize_conditioning_values(thickness_batch, uiuc_dict['uiuc_max_thickness_mean'], uiuc_dict['uiuc_max_thickness_std'])
    thickness_batch = thickness_batch + 2
    camber_batch = torch.tensor(camber_list).to(device)
    camber_batch = standardize_conditioning_values(camber_batch, uiuc_dict['uiuc_max_camber_mean'], uiuc_dict['uiuc_max_camber_std'])
    camber_batch = camber_batch + 2
    airfoils = thickness_camber_model_diffusion.sample(batch_size=len(thickness_list), conditioning=torch.cat([thickness_batch.unsqueeze(1), camber_batch.unsqueeze(1)], dim=1).to(device))
    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            y_coords = torch.cat([airfoils[i, 0], airfoils[i, 1]]).cpu().detach().numpy()
            y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[i, 1].cpu().detach().numpy()
            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, airfoil_x, s=0.1)
            ax.plot(airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f'thickness_{thickness_list[i]:.2f}_camber_{camber_list[i]:.2f}')
        plt.show()
    if pca:
        pca_1 = PCA(n_components=2)
        airfoil_matrix = airfoils.view(len(thickness_list), -1).cpu().detach().numpy()
        transformed_data = pca_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('PCA: Thickness')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('PCA: Camber')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=camber_list, cmap='viridis')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Camber Conditioning Value')
        plt.show()
    if tsne:
        tsne_1 = TSNE(n_components=2)
        transformed_data = tsne_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('TSNE')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=thickness_list, cmap='viridis')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Thickness Conditioning Value')
        plt.show()

elif model_type == 'cl':
    for cl in cl_list:
        cl = standardize_conditioning_values(torch.tensor(cl), uiuc_dict['uiuc_cl_mean'], uiuc_dict['uiuc_cl_std'])
        cl = torch.tensor(cl).unsqueeze(0).to(device) + 2
elif model_type == 'camber':
    camber_batch = torch.tensor(camber_list).to(device)
    camber_batch = standardize_conditioning_values(camber_batch, uiuc_dict['uiuc_max_camber_mean'], uiuc_dict['uiuc_max_camber_std'])
    camber_batch = camber_batch + 2
    airfoils = camber_model_diffusion.sample(batch_size=len(camber_list), conditioning=camber_batch.unsqueeze(1).to(device))

    if plot:
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            y_coords = torch.cat([airfoils[i, 0], airfoils[i, 1]]).cpu().detach().numpy()
            y_coord_upper = airfoils[i, 0].cpu().detach().numpy()
            y_coord_lower = airfoils[i, 1].cpu().detach().numpy()
            y_coords = smooth_airfoil(y_coord_lower, y_coord_upper, airfoil_x, s=0.1)
            ax.plot(airfoil_x, y_coords, color='black', linewidth=3)
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f'camber_{camber_list[i]}')
        plt.show()
    if pca:
        pca_1 = PCA(n_components=2)
        airfoil_matrix = airfoils.view(len(camber_list), -1).cpu().detach().numpy()
        transformed_data = pca_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('PCA')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=camber_list, cmap='viridis')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Camber Conditioning Value')
        plt.show()
    if tsne:
        tsne_1 = TSNE(n_components=2, perplexity=15)
        transformed_data = tsne_1.fit_transform(airfoil_matrix)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title('TSNE')
        scatter = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=camber_list, cmap='viridis')
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Camber Conditioning Value')
        plt.show()

