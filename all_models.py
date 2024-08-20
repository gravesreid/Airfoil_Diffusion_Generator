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
from MakeDatasets import *

# Save paths for data
pkl_save_path_cd = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cd.pkl'
pkl_save_path_cl = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cl.pkl'
pkl_save_path_cl_cd = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cl_cd.pkl'
pkl_save_path_all = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_all.pkl'
uiuc_pkl_path = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/uiuc_airfoils.pkl'

# Load UIUC airfoil data
if os.path.exists(uiuc_pkl_path):
    print("Loading UIUC airfoils...")
    with open(uiuc_pkl_path, 'rb') as f:
        uiuc_data = pickle.load(f)
        uiuc_coordinates_list = uiuc_data['uiuc_coordinates']
        uiuc_cl_values = uiuc_data['uiuc_cl_values']
        uiuc_cd_values = uiuc_data['uiuc_cd_values']
        uiuc_max_camber = uiuc_data['uiuc_max_camber']
        uiuc_max_thickness = uiuc_data['uiuc_max_thickness']
        uiuc_names = uiuc_data['uiuc_names']
        uiuc_fitness_mean = uiuc_data['uiuc_fitness_mean']
        uiuc_fitness_std = uiuc_data['uiuc_fitness_std']
        uiuc_cl_mean = uiuc_data['uiuc_cl_mean']
        uiuc_cl_std = uiuc_data['uiuc_cl_std']
        uiuc_cd_mean = uiuc_data['uiuc_cd_mean']
        uiuc_cd_std = uiuc_data['uiuc_cd_std']
        uiuc_max_camber_mean = uiuc_data['uiuc_max_camber_mean']
        uiuc_max_camber_std = uiuc_data['uiuc_max_camber_std']
        uiuc_max_thickness_mean = uiuc_data['uiuc_max_thickness_mean']
        uiuc_max_thickness_std = uiuc_data['uiuc_max_thickness_std']
        uiuc_max_fitness = uiuc_data['uiuc_max_fitness']
        uiuc_min_fitness = uiuc_data['uiuc_min_fitness']
        uiuc_max_cl = uiuc_data['uiuc_max_cl']
        uiuc_min_cl = uiuc_data['uiuc_min_cl']
        uiuc_max_cd = uiuc_data['uiuc_max_cd']
        uiuc_min_cd = uiuc_data['uiuc_min_cd']
        uiuc_max_camber = uiuc_data['uiuc_max_camber']
        uiuc_min_camber = uiuc_data['uiuc_min_camber']
        uiuc_max_thickness = uiuc_data['uiuc_max_thickness']
        uiuc_min_thickness = uiuc_data['uiuc_min_thickness']
else:
    uiuc_airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    uiuc_dataset = AirfoilDataset(uiuc_airfoil_path, num_points_per_side=100)
    uiuc_dataloader = DataLoader(uiuc_dataset, batch_size=1, shuffle=True)

    uiuc_cl_values = []
    uiuc_cd_values = []
    uiuc_coordinates_list = []
    uiuc_max_camber = []
    uiuc_max_thickness = []
    uiuc_names = []
    uiuc_fitness_list = []
    for i, uiuc_airfoil in enumerate(uiuc_dataloader):
        uiuc_coordinates = uiuc_airfoil['coordinates']
        uiuc_coordinates_list.append(uiuc_coordinates)
        cl = uiuc_airfoil['CL'][0]
        cd = uiuc_airfoil['CD'][0]
        uiuc_cl_values.append(cl)
        uiuc_cd_values.append(cd)
        max_camber = uiuc_airfoil['max_camber']
        uiuc_max_camber.append(max_camber)
        max_thickness = uiuc_airfoil['max_thickness']
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
    uiuc_max_camber = max(uiuc_max_camber)
    uiuc_min_camber = min(uiuc_max_camber)
    uiuc_max_thickness = max(uiuc_max_thickness)
    uiuc_min_thickness = min(uiuc_max_thickness)
    uiuc_data_to_save = {
        'uiuc_coordinates': uiuc_coordinates_list,
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
        'uiuc_max_camber': uiuc_max_camber,
        'uiuc_min_camber': uiuc_min_camber,
        'uiuc_max_thickness': uiuc_max_thickness,
        'uiuc_min_thickness': uiuc_min_thickness
    }
    with open(uiuc_pkl_path, 'wb') as f:
        pickle.dump(uiuc_data_to_save, f)

# trained model paths
vae_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
diffusion_path_cd = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cd_standardized_run_1/best_model.pt"
diffusion_path_cl = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_normalized_run_1/best_model.pt"
diffusion_path_cl_cd= "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_cd_standardized_run_1/best_model.pt"



if os.path.exists(pkl_save_path_cd):
    print("Loading saved data...")
    with open(pkl_save_path_cd, 'rb') as f:
        loaded_data = pickle.load(f)
        cd_model_latent_vectors = loaded_data['latent_vectors']
        cd_model_cd_values = loaded_data['cd_values']
        cd_model_y_coords_list = loaded_data['y_coords_list']
        cd_model_gen_coefficients = loaded_data['gen_coefficients']
        cd_model_gen_max_camber = loaded_data['gen_max_camber']
        cd_model_gen_max_thickness = loaded_data['gen_max_thickness']
        cd_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cd_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cd_model_airfoil_x = loaded_data['airfoil_x']
elif os.path.exists(pkl_save_path_cl):
    print("Loading saved data...")
    with open(pkl_save_path_cl, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_model_latent_vectors = loaded_data['latent_vectors']
        cl_model_cl_values = loaded_data['cl_values']
        cl_model_y_coords_list = loaded_data['y_coords_list']
        cl_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_model_gen_max_camber = loaded_data['gen_max_camber']
        cl_model_gen_max_thickness = loaded_data['gen_max_thickness']
        cl_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cl_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cl_model_airfoil_x = loaded_data['airfoil_x']
elif os.path.exists(pkl_save_path_cl_cd):
    print("Loading saved data...")
    with open(pkl_save_path_cl_cd, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_cd_model_latent_vectors = loaded_data['latent_vectors']
        cl_cd_model_cl_values = loaded_data['cl_values']
        cl_cd_model_cd_values = loaded_data['cd_values']
        cl_cd_model_y_coords_list = loaded_data['y_coords_list']
        cl_cd_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_cd_model_gen_max_camber = loaded_data['gen_max_camber']
        cl_cd_model_gen_max_thickness = loaded_data['gen_max_thickness']
        cl_cd_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cl_cd_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cl_cd_model_airfoil_x = loaded_data['airfoil_x']
elif os.path.exists(pkl_save_path_all):
    print("Loading saved data...")
    with open(pkl_save_path_all, 'rb') as f:
        loaded_data = pickle.load(f)
        all_model_latent_vectors = loaded_data['latent_vectors']
        all_model_cl_values = loaded_data['cl_values']
        all_model_cd_values = loaded_data['cd_values']
        all_model_y_coords_list = loaded_data['y_coords_list']
        all_model_gen_coefficients = loaded_data['gen_coefficients']
        all_model_gen_max_camber = loaded_data['gen_max_camber']
        all_model_gen_max_thickness = loaded_data['gen_max_thickness']
        all_model_conditioning_error_list = loaded_data['conditioning_error_list']
        all_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        all_model_airfoil_x = loaded_data['airfoil_x']
else:
    models_to_generate = []
    if not os.path.exists(pkl_save_path_cd):
        models_to_generate.append('cd')
    if not os.path.exists(pkl_save_path_cl):
        models_to_generate.append('cl')
    if not os.path.exists(pkl_save_path_cl_cd):
        models_to_generate.append('cl_cd')
    if not os.path.exists(pkl_save_path_all):
        models_to_generate.append('all')
    for model in models_to_generate:
        if model == 'cd':
            print("Generating airfoils with CD conditioning...")
            generate_airfoils_cd(vae_path, diffusion_path_cd, pkl_save_path_cd, uiuc_cd_values)
        elif model == 'cl':
            print("Generating airfoils with CL conditioning...")
            generate_airfoils_cl(vae_path, diffusion_path_cl, pkl_save_path_cl, uiuc_cl_values)
        elif model == 'cl_cd':
            print("Generating airfoils with CL and CD conditioning...")
            generate_airfoils_cl_cd(vae_path, diffusion_path_cl_cd, pkl_save_path_cl_cd, uiuc_cl_values, uiuc_cd_values)
        elif model == 'all':
            print("Generating airfoils with all conditioning values...")
            generate_airfoils_all(vae_path, diffusion_path_cl_cd, pkl_save_path_all, uiuc_cl_values, uiuc_cd_values)


