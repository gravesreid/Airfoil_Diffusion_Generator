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
from sklearn.metrics import r2_score


# load uiuc airfoil data
uiuc_path = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/uiuc_airfoils.pkl'

with open(uiuc_path, 'rb') as f:
    uiuc_data = pickle.load(f)
    uiuc_cl_mean = uiuc_data['uiuc_cl_mean']
    uiuc_cl_std = uiuc_data['uiuc_cl_std']
    uiuc_cd_mean = uiuc_data['uiuc_cd_mean']
    uiuc_cd_std = uiuc_data['uiuc_cd_std']
    uiuc_max_camber_mean = uiuc_data['uiuc_max_camber_mean']
    uiuc_max_camber_std = uiuc_data['uiuc_max_camber_std']
    uiuc_max_thickness_mean = uiuc_data['uiuc_max_thickness_mean']
    uiuc_max_thickness_std = uiuc_data['uiuc_max_thickness_std']
    uiuc_max_cl = uiuc_data['uiuc_max_cl']
    uiuc_max_cd = uiuc_data['uiuc_max_cd']
    uiuc_min_cl = uiuc_data['uiuc_min_cl']
    uiuc_min_cd = uiuc_data['uiuc_min_cd']
    uiuc_max_thickness = uiuc_data['uiuc_max_thickness']
    uiuc_min_thickness = uiuc_data['uiuc_min_thickness']
    uiuc_max_camber = uiuc_data['uiuc_max_camber']
    uiuc_min_camber = uiuc_data['uiuc_min_camber']



def normalize_conditioning_values(conditioning_values, min_values, max_values):
    # Normalize the conditioning values
    normalized_conditioning_values = (conditioning_values - min_values) / (max_values - min_values)
    return normalized_conditioning_values

def standardize_conditioning_values(conditioning_values, mean_values, std_values):
    # Standardize the conditioning values
    standardized_conditioning_values = (conditioning_values - mean_values) / std_values
    return standardized_conditioning_values

def unstandardize_conditioning_values(conditioning_values, mean_values, std_values):
    # Unstandardize the conditioning values
    unstandardized_conditioning_values = (conditioning_values * std_values) + mean_values
    return unstandardized_conditioning_values

def unnormalize_conditioning_values(conditioning_values, min_values, max_values):
    # Unnormalize the conditioning values
    unnormalized_conditioning_values = (conditioning_values * (max_values - min_values)) + min_values
    return unnormalized_conditioning_values

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




def generate_airfoils_cd(vae_path, diffusion_path, pkl_save_path, uiuc_cd_values, standardize=True, normalize=False):
    # Parameters
    airfoil_dim = 200
    latent_dim = 100
    n_samples = 6
    unet_dim = 12
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the trained VAE model
    vae_model = VAE(airfoil_dim, latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
    vae_model.eval()

    # Load the trained diffusion model
    diffusion_model = Unet1DConditional(unet_dim, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
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
    # Define ranges for CL and CD
    # get range of cd values from uiuc airfoils
    min_cd = min(uiuc_cd_values)
    max_cd = max(uiuc_cd_values)
    cd_range = torch.linspace(min_cd, max_cd, 250).unsqueeze(1).to(device)

    # Create a grid of CL and CD values
    n_samples = cd_range.shape[0]

    # Generate airfoils for each CL, CD pair
    latent_vectors = []
    cd_values = []

    samples_generated = 0
    batch_size = 4 # Set the batch size

    y_coords_list = []
    airfoil_list = []
    gen_coefficients = []
    gen_max_camber = []
    gen_max_thickness = []
    conditioning_error_list = []
    conditioning_difference_list = []
    for conditioning_values in cd_range:
        if standardize:
            standardized_values = standardize_conditioning_values(conditioning_values, uiuc_cd_mean, uiuc_cd_std)
            standardized_values = standardized_values.repeat(batch_size, 1)
            generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)
        elif normalize:
            normalized_values = normalize_conditioning_values(conditioning_values, uiuc_min_cd, uiuc_max_cd)
            normalized_values = normalized_values.repeat(batch_size, 1)
            generated_y = diffusion.sample(batch_size=batch_size, conditioning=normalized_values)
        
        for i in range(batch_size):
            # get y_coords into one channel
            y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
            
            # Get the latent vector using the VAE encoder
            latent_vector = vae_model.enc(y_coords)
            latent_vector = latent_vector[0].detach().cpu().numpy()
            latent_vector = torch.tensor(latent_vector)
            latent_vectors.append(latent_vector)
            
            # Store the conditioning values for each sample
            cd_values.append(conditioning_values[0].detach().cpu())

            # store y_coords for plotting andCreate an airfoil object
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            y_coords_list.append(y_coords)
            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name = f'Generated Airfoil {i+1}',
                coordinates = coordinates
            )
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            cd = coefficients['CD'][0]
            conditioning_difference = cd - conditioning_values[0]
            conditioning_difference_list.append(conditioning_difference)
            conditioning_error = torch.abs(cd - conditioning_values[0])/torch.abs(conditioning_values[0])
            conditioning_error_list.append(conditioning_error)
            gen_coefficients.append(coefficients)
            max_camber = airfoil.max_camber()
            gen_max_camber.append(max_camber)
            max_thickness = airfoil.max_thickness()
            gen_max_thickness.append(max_thickness)

        samples_generated += batch_size
        print(f"Generated airfoil batch: {samples_generated}/{n_samples * batch_size}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cd_values': cd_values,
        'y_coords_list': y_coords_list,
        'gen_coefficients': gen_coefficients,
        'gen_max_camber': gen_max_camber,
        'gen_max_thickness': gen_max_thickness,
        'conditioning_error_list': conditioning_error_list,
        'conditioning_difference_list': conditioning_difference_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')


def generate_airfoils_cl(vae_path, diffusion_path, pkl_save_path, uiuc_cl_values, standardize=True, normalize=False):
    # Parameters
    airfoil_dim = 200
    latent_dim = 100
    n_samples = 6
    unet_dim = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained VAE model
    vae_model = VAE(airfoil_dim, latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
    vae_model.eval()

    # Load the trained diffusion model
    diffusion_model = Unet1DConditional(unet_dim, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
    diffusion_model.eval()
    diffusion = GaussianDiffusion1D(diffusion_model, seq_length=latent_dim).to(device)

    # Initialize dataset
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    min_cl = min(uiuc_cl_values)
    max_cl = max(uiuc_cl_values)
    cl_range = torch.linspace(min_cl, max_cl, 250).unsqueeze(1).to(device)

    batch_size = 4
    samples_generated = 0
    latent_vectors, cl_values, y_coords_list = [], [], []

    for conditioning_values in cl_range:
        if standardize:
            standardized_values = standardize_conditioning_values(conditioning_values, uiuc_cl_mean, uiuc_cl_std)
            standardized_values = standardized_values.repeat(batch_size, 1)
            generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)
        elif normalize:
            normalized_values = normalize_conditioning_values(conditioning_values, uiuc_min_cl, uiuc_max_cl)
            normalized_values = normalized_values.repeat(batch_size, 1)
            generated_y = diffusion.sample(batch_size=batch_size, conditioning=normalized_values)

        for i in range(batch_size):
            y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
            latent_vector = vae_model.enc(y_coords)
            latent_vector = latent_vector[0].detach().cpu().numpy()
            latent_vectors.append(torch.tensor(latent_vector))

            cl_values.append(conditioning_values[0].detach().cpu())
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            y_coords_list.append(y_coords)

        samples_generated += batch_size
        print(f"Generated airfoil batch: {samples_generated}/{len(cl_range) * batch_size}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_values': cl_values,
        'y_coords_list': y_coords_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')

def generate_airfoils_cl_cd(vae_path, diffusion_path, pkl_save_path, uiuc_cl_values, uiuc_cd_values, standardize=True, normalize=False):
    # Parameters
    airfoil_dim = 200
    latent_dim = 100
    n_samples = 6
    unet_dim = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained VAE model
    vae_model = VAE(airfoil_dim, latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
    vae_model.eval()

    # Load the trained diffusion model
    diffusion_model = Unet1DConditional(unet_dim, cond_dim=2, channels=2, dim_mults=(1,2,4)).to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
    diffusion_model.eval()
    diffusion = GaussianDiffusion1D(diffusion_model, seq_length=latent_dim).to(device)

    # Initialize dataset
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    cl_range = torch.linspace(min(uiuc_cl_values), max(uiuc_cl_values), 125).unsqueeze(1).to(device)
    cd_range = torch.linspace(min(uiuc_cd_values), max(uiuc_cd_values), 125).unsqueeze(1).to(device)

    latent_vectors, cl_values, cd_values, y_coords_list = [], [], [], []
    batch_size = 4
    samples_generated = 0

    for cl_value in cl_range:
        for cd_value in cd_range:
            conditioning_values = torch.cat([cl_value, cd_value]).unsqueeze(0).to(device)
            
            if standardize:
                standardized_values = standardize_conditioning_values(conditioning_values, [uiuc_cl_mean, uiuc_cd_mean], [uiuc_cl_std, uiuc_cd_std])
                standardized_values = standardized_values.repeat(batch_size, 1)
                generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)
            elif normalize:
                normalized_values = normalize_conditioning_values(conditioning_values, [uiuc_min_cl, uiuc_min_cd], [uiuc_max_cl, uiuc_max_cd])
                normalized_values = normalized_values.repeat(batch_size, 1)
                generated_y = diffusion.sample(batch_size=batch_size, conditioning=normalized_values)

            for i in range(batch_size):
                y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
                latent_vector = vae_model.enc(y_coords)
                latent_vector = latent_vector[0].detach().cpu().numpy()
                latent_vectors.append(torch.tensor(latent_vector))

                cl_values.append(cl_value[0].detach().cpu())
                cd_values.append(cd_value[0].detach().cpu())
                y_coords = y_coords.detach().cpu().numpy()
                y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
                y_coords_list.append(y_coords)

            samples_generated += batch_size
            print(f"Generated airfoil batch: {samples_generated}/{len(cl_range) * len(cd_range) * batch_size}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_values': cl_values,
        'cd_values': cd_values,
        'y_coords_list': y_coords_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')

def generate_airfoils_all(vae_path, diffusion_path, pkl_save_path, uiuc_cl_values, uiuc_cd_values, uiuc_thickness_values, uiuc_camber_values, standardize=True, normalize=False):
    # Parameters
    airfoil_dim = 200
    latent_dim = 100
    n_samples = 6
    unet_dim = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained VAE model
    vae_model = VAE(airfoil_dim, latent_dim).to(device)
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
    vae_model.eval()

    # Load the trained diffusion model
    diffusion_model = Unet1DConditional(unet_dim, cond_dim=4, channels=2, dim_mults=(1,2,4)).to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
    diffusion_model.eval()
    diffusion = GaussianDiffusion1D(diffusion_model, seq_length=latent_dim).to(device)

    # Initialize dataset
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    cl_range = torch.linspace(min(uiuc_cl_values), max(uiuc_cl_values), 100).unsqueeze(1).to(device)
    cd_range = torch.linspace(min(uiuc_cd_values), max(uiuc_cd_values), 100).unsqueeze(1).to(device)
    thickness_range = torch.linspace(min(uiuc_thickness_values), max(uiuc_thickness_values), 100).unsqueeze(1).to(device)
    camber_range = torch.linspace(min(uiuc_camber_values), max(uiuc_camber_values), 100).unsqueeze(1).to(device)

    latent_vectors, cl_values, cd_values, thickness_values, camber_values, y_coords_list = [], [], [], [], [], []
    batch_size = 4
    samples_generated = 0

    for cl_value in cl_range:
        for cd_value in cd_range:
            for thickness_value in thickness_range:
                for camber_value in camber_range:
                    conditioning_values = torch.cat([cl_value, cd_value, thickness_value, camber_value]).unsqueeze(0).to(device)
                    
                    if standardize:
                        standardized_values = standardize_conditioning_values(conditioning_values,
                            [uiuc_cl_mean, uiuc_cd_mean, uiuc_max_thickness_mean, uiuc_max_camber_mean],
                            [uiuc_cl_std, uiuc_cd_std, uiuc_max_thickness_std, uiuc_max_camber_std])
                        standardized_values = standardized_values.repeat(batch_size, 1)
                        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)
                    elif normalize:
                        normalized_values = normalize_conditioning_values(conditioning_values,
                            [uiuc_min_cl, uiuc_min_cd, uiuc_min_thickness, uiuc_min_camber],
                            [uiuc_max_cl, uiuc_max_cd, uiuc_max_thickness, uiuc_max_camber])
                        normalized_values = normalized_values.repeat(batch_size, 1)
                        generated_y = diffusion.sample(batch_size=batch_size, conditioning=normalized_values)

                    for i in range(batch_size):
                        y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
                        latent_vector = vae_model.enc(y_coords)
                        latent_vector = latent_vector[0].detach().cpu().numpy()
                        latent_vectors.append(torch.tensor(latent_vector))

                        cl_values.append(cl_value[0].detach().cpu())
                        cd_values.append(cd_value[0].detach().cpu())
                        thickness_values.append(thickness_value[0].detach().cpu())
                        camber_values.append(camber_value[0].detach().cpu())
                        y_coords = y_coords.detach().cpu().numpy()
                        y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
                        y_coords_list.append(y_coords)

                    samples_generated += batch_size
                    print(f"Generated airfoil batch: {samples_generated}/{len(cl_range) * len(cd_range) * len(thickness_range) * len(camber_range) * batch_size}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_values': cl_values,
        'cd_values': cd_values,
        'thickness_values': thickness_values,
        'camber_values': camber_values,
        'y_coords_list': y_coords_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')
