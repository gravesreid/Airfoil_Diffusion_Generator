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
uiuc_path = 'uiuc_airfoils.pkl'
if not os.path.exists(uiuc_path):
    uiuc_airfoil_path = 'coord_seligFmt'
    uiuc_dataset = AirfoilDataset(uiuc_airfoil_path, num_points_per_side=100)
    uiuc_dataloader = DataLoader(uiuc_dataset, batch_size=1, shuffle=True)

    uiuc_cl_values = []
    uiuc_cd_values = []
    uiuc_coordinates_list = []
    uiuc_training_coordinates_list = []
    uiuc_max_camber = []
    uiuc_max_thickness = []
    uiuc_names = []
    uiuc_fitness_list = []
    for i, uiuc_airfoil in enumerate(uiuc_dataloader):
        uiuc_coordinates = uiuc_airfoil['coordinates']
        uiuc_training_coordinates = uiuc_airfoil['train_coords_y']
        uiuc_training_coordinates_list.append(uiuc_training_coordinates)
        uiuc_coordinates_list.append(uiuc_coordinates)
        cl = uiuc_airfoil['CL'][0]
        cd = uiuc_airfoil['CD'][0]
        uiuc_cl_values.append(cl)
        uiuc_cd_values.append(cd)
        max_camber = uiuc_airfoil['max_camber']
        uiuc_max_camber.append(max_camber)
        max_thickness = uiuc_airfoil['max_thickness']
        print(f'max_thickness: {max_thickness}')
        uiuc_max_thickness.append(max_thickness)
        name = uiuc_airfoil['name']
        uiuc_names.append(name)
        fitness = cl/cd
        uiuc_fitness_list.append(fitness)
    # find mean, standard deviation of uiuc fitness values, cl, cd, max_camber, max_thickness, then save all to pkl file
    uiuc_fitness_mean = np.mean(uiuc_fitness_list)
    uiuc_fitness_std = np.std(uiuc_fitness_list)
    uiuc_cl_mean = np.mean(uiuc_cl_values)
    uiuc_cl_std = np.std(uiuc_cl_values)
    uiuc_cd_mean = np.mean(uiuc_cd_values)
    uiuc_cd_std = np.std(uiuc_cd_values)
    uiuc_max_camber_mean = np.mean(uiuc_max_camber)
    uiuc_max_camber_std = np.std(uiuc_max_camber)
    uiuc_max_thickness_mean = np.mean(uiuc_max_thickness)
    uiuc_max_thickness_std = np.std(uiuc_max_thickness)
    uiuc_max_fitness = max(uiuc_fitness_list)
    uiuc_min_fitness = min(uiuc_fitness_list)
    uiuc_max_cl = max(uiuc_cl_values)
    uiuc_min_cl = min(uiuc_cl_values)
    uiuc_max_cd = max(uiuc_cd_values)
    uiuc_min_cd = min(uiuc_cd_values)
    uiuc_max_max_camber = max(uiuc_max_camber)
    uiuc_min_max_camber = min(uiuc_max_camber)
    uiuc_max_max_thickness = max(uiuc_max_thickness)
    uiuc_min_max_thickness = min(uiuc_max_thickness)
    uiuc_data_to_save = {
        'uiuc_coordinates': uiuc_coordinates_list,
        'uiuc_training_coordinates': uiuc_training_coordinates_list,
        'uiuc_cl_values': uiuc_cl_values,
        'uiuc_cd_values': uiuc_cd_values,
        'uiuc_max_camber': uiuc_max_camber,
        'uiuc_max_thickness': uiuc_max_thickness,
        'uiuc_names': uiuc_names,
        'uiuc_fitness_mean': uiuc_fitness_mean,
        'uiuc_fitness_std': uiuc_fitness_std,
        'uiuc_cl_mean': uiuc_cl_mean,
        'uiuc_cl_std': uiuc_cl_std,
        'uiuc_cd_mean': uiuc_cd_mean,
        'uiuc_cd_std': uiuc_cd_std,
        'uiuc_max_camber_mean': uiuc_max_camber_mean,
        'uiuc_max_camber_std': uiuc_max_camber_std,
        'uiuc_max_thickness_mean': uiuc_max_thickness_mean,
        'uiuc_max_thickness_std': uiuc_max_thickness_std,
        'uiuc_max_fitness': uiuc_max_fitness,
        'uiuc_min_fitness': uiuc_min_fitness,
        'uiuc_max_cl': uiuc_max_cl,
        'uiuc_min_cl': uiuc_min_cl,
        'uiuc_max_cd': uiuc_max_cd,
        'uiuc_min_cd': uiuc_min_cd,
        'uiuc_max_camber': uiuc_max_max_camber,
        'uiuc_min_camber': uiuc_min_max_camber,
        'uiuc_max_thickness': uiuc_max_max_thickness,
        'uiuc_min_thickness': uiuc_min_max_thickness
    }
    with open(uiuc_path, 'wb') as f:
        pickle.dump(uiuc_data_to_save, f)
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
    # Ensure all tensors are on the same device as conditioning_values
    device = conditioning_values.device
    min_values = min_values.to(device)
    max_values = max_values.to(device)
    data_type = conditioning_values.dtype
    min_values = min_values.type(data_type)
    max_values = max_values.type(data_type)
    
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
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    min_cd = min(uiuc_cd_values)
    max_cd = max(uiuc_cd_values)
    cd_range = torch.linspace(min_cd, max_cd, 3000).unsqueeze(1).to(device)

    batch_size = 1000
    num_batches = len(cd_range) // batch_size

    latent_vectors = []
    cd_conditioning_values = []
    cd_eval_values = []
    cl_eval_values = []
    maximum_thickness_list = []
    maximimum_camber_list = []
    y_coords_list = []
    gen_coefficients = []
    conditioning_error_list = []
    conditioning_difference_list = []
    conditioning_MAE_list = []


    for i in range(0, len(cd_range), batch_size):
        batch_conditioning_values = cd_range[i:i + batch_size]
        if standardize:
            standardized_values = standardize_conditioning_values(batch_conditioning_values, uiuc_cd_mean, uiuc_cd_std)
        elif normalize:
            standardized_values = normalize_conditioning_values(batch_conditioning_values, uiuc_min_cd, uiuc_max_cd)

        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)

        for j in range(batch_size):
            y_coords = torch.cat([generated_y[j, 0, :], generated_y[j, 1, :]])
            latent_vector = vae_model.enc(y_coords)[0].detach().cpu().numpy()
            latent_vectors.append(torch.tensor(latent_vector))

            cd_conditioning_values.append(batch_conditioning_values[j].detach().cpu())
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name=f'CD_Generated Airfoil {i * batch_size + j + 1}',
                coordinates=coordinates
            )
            airfoil = airfoil.repanel(n_points_per_side=100)
            airfoil = airfoil.normalize()
            y_coords_list.append(airfoil.coordinates[:, 1])
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            cd = coefficients['CD'][0]
            cl = coefficients['CL'][0]
            cd_eval_values.append(cd)
            cl_eval_values.append(cl)
            maximum_thickness = airfoil.max_thickness()
            maximum_thickness_list.append(maximum_thickness)
            maximum_camber = airfoil.max_camber()
            maximimum_camber_list.append(maximum_camber)
            conditioning_difference = cd - batch_conditioning_values[j]
            conditioning_difference_list.append(conditioning_difference)
            conditioning_error = torch.abs(cd - batch_conditioning_values[j]) / torch.abs(batch_conditioning_values[j])
            conditioning_error_list.append(conditioning_error)
            conditioning_MAE = torch.abs(cd - batch_conditioning_values[j])
            conditioning_MAE_list.append(conditioning_MAE)
            gen_coefficients.append(coefficients)

        print(f"Generated airfoil batch: {(i + batch_size)}/{len(cd_range)}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cd_conditioning_values': cd_conditioning_values,
        'cd_eval_values': cd_eval_values,
        'cl_eval_values': cl_eval_values,
        'maximum_thickness_list': maximum_thickness_list,
        'maximum_camber_list': maximimum_camber_list,
        'y_coords_list': y_coords_list,
        'gen_coefficients': gen_coefficients,
        'conditioning_error_list': conditioning_error_list,
        'conditioning_difference_list': conditioning_difference_list,
        'conditioning_MAE_list': conditioning_MAE_list,
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
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    min_cl = min(uiuc_cl_values)
    max_cl = max(uiuc_cl_values)
    cl_range = torch.linspace(min_cl, max_cl*2, 3000).unsqueeze(1).to(device)

    batch_size = 1000
    num_batches = len(cl_range) // batch_size

    latent_vectors = []
    cl_conditioning_values = []
    cl_eval_values = []
    cd_eval_values = []
    y_coords_list = []
    gen_coefficients = []
    conditioning_error_list = []
    conditioning_difference_list = []
    conditioning_MAE_list = []
    maximum_thickness_list = []
    maximimum_camber_list = []

    for i in range(0, len(cl_range), batch_size):
        batch_conditioning_values = cl_range[i:i + batch_size]
        if standardize:
            standardized_values = standardize_conditioning_values(batch_conditioning_values, uiuc_cl_mean, uiuc_cl_std)
        elif normalize:
            standardized_values = normalize_conditioning_values(batch_conditioning_values, uiuc_min_cl, uiuc_max_cl)
        
        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)

        for j in range(batch_size):
            y_coords = torch.cat([generated_y[j, 0, :], generated_y[j, 1, :]])
            latent_vector = vae_model.enc(y_coords)[0].detach().cpu().numpy()
            latent_vectors.append(torch.tensor(latent_vector))

            cl_conditioning_values.append(batch_conditioning_values[j].detach().cpu())
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name=f'CL_Generated Airfoil {i * batch_size + j + 1}',
                coordinates=coordinates
            )
            airfoil = airfoil.repanel(n_points_per_side=100)
            airfoil = airfoil.normalize()
            y_coords_list.append(airfoil.coordinates[:, 1])
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            cl = coefficients['CL'][0]
            cl_eval_values.append(cl)
            cd = coefficients['CD'][0]
            cd_eval_values.append(cd)
            conditioning_difference = cl - batch_conditioning_values[j]
            conditioning_difference_list.append(conditioning_difference)
            conditioning_error = torch.abs(cl - batch_conditioning_values[j]) / torch.abs(batch_conditioning_values[j])
            conditioning_error_list.append(conditioning_error)
            conditioning_MAE = torch.abs(cl - batch_conditioning_values[j])
            conditioning_MAE_list.append(conditioning_MAE)
            gen_coefficients.append(coefficients)
            maximum_thickness = airfoil.max_thickness()
            maximum_thickness_list.append(maximum_thickness)
            maximum_camber = airfoil.max_camber()
            maximimum_camber_list.append(maximum_camber)

        print(f"Generated airfoil batch: {(i + batch_size)}/{len(cl_range)}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_conditioning_values': cl_conditioning_values,
        'cl_eval_values': cl_eval_values,
        'cd_eval_values': cd_eval_values,
        'maximum_thickness_list': maximum_thickness_list,
        'maximum_camber_list': maximimum_camber_list,
        'y_coords_list': y_coords_list,
        'gen_coefficients': gen_coefficients,
        'conditioning_error_list': conditioning_error_list,
        'conditioning_difference_list': conditioning_difference_list,
        'conditioning_MAE_list': conditioning_MAE_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')

def generate_airfoils_cl_thickness(vae_path, diffusion_path, pkl_save_path, uiuc_cl_values, uiuc_thickness_values, standardize=True, normalize=False):
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
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    cl_range = torch.linspace(min(uiuc_cl_values), max(uiuc_cl_values)*2, 100).unsqueeze(1).to(device)
    thickness_range = torch.linspace(min(uiuc_thickness_values), max(uiuc_thickness_values)*4, 100).unsqueeze(1).to(device)

    # Create all combinations of CL and thickness
    cl_thickness_combinations = torch.cartesian_prod(cl_range.squeeze(), thickness_range.squeeze()).to(device)

    # Prepare storage for results
    latent_vectors, cl_conditioning_values, thickness_conditioning_values = [], [], []
    cl_eval_values, thickness_eval_values, gen_coefficients = [], [], []
    cl_difference_list, thickness_difference_list, cl_error_list, thickness_error_list = [], [], [], []
    y_coords_list = []
    cd_eval_values = []

    batch_size = 1000
    num_combinations = cl_thickness_combinations.shape[0]
    num_batches = num_combinations // batch_size

    for i in range(0, num_combinations, batch_size):
        batch_combinations = cl_thickness_combinations[i:i + batch_size]
        batch_cl_values = batch_combinations[:, 0].unsqueeze(1)
        batch_thickness_values = batch_combinations[:, 1].unsqueeze(1)

        if standardize:
            standardized_values_cl = standardize_conditioning_values(batch_cl_values, uiuc_cl_mean, uiuc_cl_std)
            standardized_values_thickness = standardize_conditioning_values(batch_thickness_values, uiuc_max_thickness_mean, uiuc_max_thickness_std)
            standardized_values = torch.cat([standardized_values_cl, standardized_values_thickness], dim=1)
        elif normalize:
            normalized_values_cl = normalize_conditioning_values(batch_cl_values, uiuc_min_cl, uiuc_max_cl)
            normalized_values_thickness = normalize_conditioning_values(batch_thickness_values, uiuc_min_thickness, uiuc_max_thickness)
            normalized_values = torch.cat([normalized_values_cl, normalized_values_thickness], dim=1)

        # Generate airfoils using diffusion model
        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)

        for j in range(batch_size):
            y_coords = torch.cat([generated_y[j, 0, :], generated_y[j, 1, :]])
            latent_vector = vae_model.enc(y_coords)[0].detach().cpu().numpy()
            latent_vectors.append(torch.tensor(latent_vector))

            cl_conditioning_values.append(batch_cl_values[j].detach().cpu())
            thickness_conditioning_values.append(batch_thickness_values[j].detach().cpu())
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            y_coords_list.append(y_coords)

            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name=f'Cl_Thickness_Generated Airfoil {i * batch_size + j + 1}',
                coordinates=coordinates
            )
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            gen_coefficients.append(coefficients)

            cl = coefficients['CL'][0]
            cl_eval_values.append(cl)
            cd = coefficients['CD'][0]
            cd_eval_values.append(cd)
            maximum_thickness = airfoil.max_thickness()
            thickness_eval_values.append(maximum_thickness)
            cl_difference = cl - batch_cl_values[j]
            thickness_difference = maximum_thickness - batch_thickness_values[j]

            cl_difference_list.append(cl_difference)
            thickness_difference_list.append(thickness_difference)

            cl_error = torch.abs(cl - batch_cl_values[j]) / torch.abs(batch_cl_values[j])
            thickness_error = torch.abs(maximum_thickness - batch_thickness_values[j]) / torch.abs(batch_thickness_values[j])
            cl_error_list.append(cl_error)
            thickness_error_list.append(thickness_error)

        print(f"Generated airfoil batch: {i + batch_size}/{num_combinations}")

    # Save the generated data
    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_conditioning_values': cl_conditioning_values,
        'thickness_conditioning_values': thickness_conditioning_values,
        'cl_eval_values': cl_eval_values,
        'cd_eval_values': cd_eval_values,
        'thickness_eval_values': thickness_eval_values,
        'gen_coefficients': gen_coefficients,
        'cl_difference_list': cl_difference_list,
        'thickness_difference_list': thickness_difference_list,
        'cl_error_list': cl_error_list,
        'thickness_error_list': thickness_error_list,
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
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    cl_range = torch.linspace(min(uiuc_cl_values), max(uiuc_cl_values), 1000).unsqueeze(1).to(device)
    cd_range = torch.linspace(min(uiuc_cd_values), max(uiuc_cd_values), 1000).unsqueeze(1).to(device)

    # Create all combinations of CL and CD
    cl_cd_combinations = torch.cartesian_prod(cl_range.squeeze(), cd_range.squeeze()).to(device)

    # Prepare storage for results
    latent_vectors, cl_conditioning_values, cd_conditioning_values = [], [], []
    cl_eval_values, cd_eval_values, gen_coefficients = [], [], []
    cl_difference_list, cd_difference_list, cl_error_list, cd_error_list = [], [], [], []
    y_coords_list = []

    batch_size = 1000
    num_combinations = cl_cd_combinations.shape[0]
    num_batches = num_combinations // batch_size

    for i in range(0, num_combinations, batch_size):
        batch_combinations = cl_cd_combinations[i:i + batch_size]
        batch_cl_values = batch_combinations[:, 0].unsqueeze(1)
        batch_cd_values = batch_combinations[:, 1].unsqueeze(1)
        
        if standardize:
            standardized_values_cl = standardize_conditioning_values(batch_cl_values, uiuc_cl_mean, uiuc_cl_std)
            standardized_values_cd = standardize_conditioning_values(batch_cd_values, uiuc_cd_mean, uiuc_cd_std)
            standardized_values = torch.cat([standardized_values_cl, standardized_values_cd], dim=1)
        elif normalize:
            normalized_values_cl = normalize_conditioning_values(batch_cl_values, uiuc_min_cl, uiuc_max_cl)
            normalized_values_cd = normalize_conditioning_values(batch_cd_values, uiuc_min_cd, uiuc_max_cd)
            normalized_values = torch.cat([normalized_values_cl, normalized_values_cd], dim=1)

        # Generate airfoils using diffusion model
        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)

        for j in range(batch_size):
            y_coords = torch.cat([generated_y[j, 0, :], generated_y[j, 1, :]])
            latent_vector = vae_model.enc(y_coords).detach().cpu().numpy()
            latent_vectors.append(torch.tensor(latent_vector))

            cl_conditioning_values.append(batch_cl_values[j].detach().cpu())
            cd_conditioning_values.append(batch_cd_values[j].detach().cpu())
            y_coords = y_coords.detach().cpu().numpy()
            y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
            y_coords_list.append(y_coords)

            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name=f'Cl_Cd_Generated Airfoil {i * batch_size + j + 1}',
                coordinates=coordinates
            )
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            gen_coefficients.append(coefficients)

            cl = coefficients['CL'][0]
            cl_eval_values.append(cl)
            cd = coefficients['CD'][0]
            cd_eval_values.append(cd)

            cl_difference = cl - batch_cl_values[j]
            cd_difference = cd - batch_cd_values[j]
            cl_difference_list.append(cl_difference)
            cd_difference_list.append(cd_difference)

            cl_error = torch.abs(cl - batch_cl_values[j]) / torch.abs(batch_cl_values[j])
            cd_error = torch.abs(cd - batch_cd_values[j]) / torch.abs(batch_cd_values[j])
            cl_error_list.append(cl_error)
            cd_error_list.append(cd_error)

        print(f"Generated airfoil batch: {i + batch_size}/{num_combinations}")

    # Save the generated data
    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_conditioning_values': cl_conditioning_values,
        'cd_conditioning_values': cd_conditioning_values,
        'cl_eval_values': cl_eval_values,
        'cd_eval_values': cd_eval_values,
        'gen_coefficients': gen_coefficients,
        'cl_difference_list': cl_difference_list,
        'cd_difference_list': cd_difference_list,
        'cl_error_list': cl_error_list,
        'cd_error_list': cd_error_list,
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
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    airfoil_x = dataset.get_x()

    cl_range = torch.linspace(min(uiuc_cl_values), max(uiuc_cl_values), 4).unsqueeze(1).to(device)
    cd_range = torch.linspace(min(uiuc_cd_values), max(uiuc_cd_values), 4).unsqueeze(1).to(device)
    thickness_range = torch.linspace(min(uiuc_thickness_values), max(uiuc_thickness_values), 4).unsqueeze(1).to(device)
    camber_range = torch.linspace(min(uiuc_camber_values), max(uiuc_camber_values), 4).unsqueeze(1).to(device)

    latent_vectors = []
    cl_conditioning_values = []
    cd_conditioning_values = []
    thickness_conditioning_values = []
    camber_conditioning_values = []
    y_coords_list = []
    cl_eval_values = []
    cd_eval_values = []
    thickness_eval_values = []
    camber_eval_values = []
    gen_coefficients = []
    cl_difference_list = []
    cd_difference_list = []
    thickness_difference_list = []
    camber_difference_list = []
    cl_error_list = []
    cd_error_list = []
    thickness_error_list = []
    camber_error_list = []
    batch_size = 4
    samples_generated = 0

    for cl_value in cl_range:
        for cd_value in cd_range:
            for thickness_value in thickness_range:
                for camber_value in camber_range:
                    if standardize:
                        standardized_values_cl = standardize_conditioning_values(cl_value, uiuc_cl_mean, uiuc_cl_std)
                        standardized_values_cd = standardize_conditioning_values(cd_value, uiuc_cd_mean, uiuc_cd_std)
                        standardized_values_thickness = standardize_conditioning_values(thickness_value, uiuc_max_thickness_mean, uiuc_max_thickness_std)
                        standardized_values_camber = standardize_conditioning_values(camber_value, uiuc_max_camber_mean, uiuc_max_camber_std)
                        standardized_values = torch.cat([standardized_values_cl, standardized_values_cd, standardized_values_thickness, standardized_values_camber]).unsqueeze(0)
                        standardized_values = standardized_values.repeat(batch_size, 1)
                        generated_y = diffusion.sample(batch_size=batch_size, conditioning=standardized_values)
                    elif normalize:
                        normalized_values_cl = normalize_conditioning_values(cl_value, uiuc_min_cl, uiuc_max_cl)
                        normalized_values_cd = normalize_conditioning_values(cd_value, uiuc_min_cd, uiuc_max_cd)
                        normalized_values_thickness = normalize_conditioning_values(thickness_value, uiuc_min_thickness, uiuc_max_thickness)
                        normalized_values_camber = normalize_conditioning_values(camber_value, uiuc_min_camber, uiuc_max_camber)
                        normalized_values = torch.cat([normalized_values_cl, normalized_values_cd, normalized_values_thickness, normalized_values_camber]).unsqueeze(0)
                        normalized_values = normalized_values.repeat(batch_size, 1)
                        generated_y = diffusion.sample(batch_size=batch_size, conditioning=normalized_values)

                    for i in range(batch_size):
                        y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
                        latent_vector = vae_model.enc(y_coords)
                        latent_vector = latent_vector[0].detach().cpu().numpy()
                        latent_vectors.append(torch.tensor(latent_vector))

                        cl_conditioning_values.append(cl_value[0].detach().cpu())
                        cd_conditioning_values.append(cd_value[0].detach().cpu())
                        thickness_conditioning_values.append(thickness_value[0].detach().cpu())
                        camber_conditioning_values.append(camber_value[0].detach().cpu())
                        y_coords = y_coords.detach().cpu().numpy()
                        y_coords = smooth_y_coords(y_coords, method='gaussian', window_size=3, sigma=3.0)
                        y_coords_list.append(y_coords)
                        coordinates = np.vstack([airfoil_x, y_coords]).T
                        airfoil = asb.Airfoil(
                            name = f'All_Generated Airfoil {i+1}',
                            coordinates = coordinates
                        )
                        coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
                        gen_coefficients.append(coefficients)
                        cl = coefficients['CL'][0]
                        cl_eval_values.append(cl)
                        cd = coefficients['CD'][0]
                        cd_eval_values.append(cd)
                        thickness = airfoil.max_thickness()
                        thickness_eval_values.append(thickness)
                        camber = airfoil.max_camber()
                        camber_eval_values.append(camber)
                        cl_difference = cl - cl_value[0]
                        cl_difference_list.append(cl_difference)
                        cd_difference = cd - cd_value[0]
                        cd_difference_list.append(cd_difference)
                        thickness_difference = thickness - thickness_value
                        thickness_difference_list.append(thickness_difference)
                        camber_difference = camber - camber_value
                        camber_difference_list.append(camber_difference)
                        cl_error = torch.abs(cl - cl_value[0])/torch.abs(cl_value[0])
                        cl_error_list.append(cl_error)
                        cd_error = torch.abs(cd - cd_value[0])/torch.abs(cd_value[0])
                        cd_error_list.append(cd_error)
                        thickness_error = torch.abs(thickness - thickness_value)/torch.abs(thickness_value)
                        thickness_error_list.append(thickness_error)
                        camber_error = torch.abs(camber - camber_value)/torch.abs(camber_value)
                        camber_error_list.append(camber_error)


                    samples_generated += batch_size
                    print(f"Generated airfoil batch: {samples_generated}/{len(cl_range) * len(cd_range) * len(thickness_range) * len(camber_range) * batch_size}")

    data_to_save = {
        'latent_vectors': latent_vectors,
        'cl_conditioning_values': cl_conditioning_values,
        'cd_conditioning_values': cd_conditioning_values,
        'thickness_conditioning_values': thickness_conditioning_values,
        'camber_conditioning_values': camber_conditioning_values,
        'cl_eval_values': cl_eval_values,
        'cd_eval_values': cd_eval_values,
        'thickness_eval_values': thickness_eval_values,
        'camber_eval_values': camber_eval_values,
        'gen_coefficients': gen_coefficients,
        'cl_difference_list': cl_difference_list,
        'cd_difference_list': cd_difference_list,
        'thickness_difference_list': thickness_difference_list,
        'camber_difference_list': camber_difference_list,
        'cl_error_list': cl_error_list,
        'cd_error_list': cd_error_list,
        'thickness_error_list': thickness_error_list,
        'camber_error_list': camber_error_list,
        'y_coords_list': y_coords_list,
        'airfoil_x': airfoil_x
    }
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f'Generated data saved to {pkl_save_path}')
