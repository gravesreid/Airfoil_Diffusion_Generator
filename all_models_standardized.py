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
from MakeDatasets import generate_airfoils_cd, generate_airfoils_cl, generate_airfoils_cl_cd, generate_airfoils_all, generate_airfoils_cl_thickness
from tqdm import tqdm
# Save paths for data
pkl_save_path_cd = 'gen_airfoils_cd_standardized.pkl'
pkl_save_path_cl = 'gen_airfoils_cl_standardized.pkl'
pkl_save_path_cl_thickness = 'gen_airfoils_cl_thickness_standardized.pkl'
pkl_save_path_cl_cd = 'gen_airfoils_cl_cd_standardized.pkl'
pkl_save_path_all = 'gen_airfoils_all_standardized.pkl'
uiuc_pkl_path = 'uiuc_airfoils.pkl'

# Load UIUC airfoil data
if os.path.exists(uiuc_pkl_path):
    print("Loading UIUC airfoils...")
    with open(uiuc_pkl_path, 'rb') as f:
        uiuc_data = pickle.load(f)
        uiuc_coordinates_list = uiuc_data['uiuc_coordinates']
        uiuc_training_coordinates = uiuc_data['uiuc_training_coordinates']
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
        'uiuc_max_camber': uiuc_max_camber,
        'uiuc_min_camber': uiuc_min_camber,
        'uiuc_max_thickness': uiuc_max_thickness,
        'uiuc_min_thickness': uiuc_min_thickness
    }
    with open(uiuc_pkl_path, 'wb') as f:
        pickle.dump(uiuc_data_to_save, f)

def print_uiuc_stats():
    print(f"UIUC Fitness Mean: {uiuc_fitness_mean:.4f}, UIUC Fitness Std: {uiuc_fitness_std:.4f}")
    print(f"UIUC CL Mean: {uiuc_cl_mean:.4f}, UIUC CL Std: {uiuc_cl_std:.4f}")
    print(f"UIUC CD Mean: {uiuc_cd_mean:.4f}, UIUC CD Std: {uiuc_cd_std:.4f}")
    print(f"UIUC Max Camber Mean: {uiuc_max_camber_mean:.4f}, UIUC Max Camber Std: {uiuc_max_camber_std:.4f}")
    print(f"UIUC Max Thickness Mean: {uiuc_max_thickness_mean:.4f}, UIUC Max Thickness Std: {uiuc_max_thickness_std:.4f}")
    print(f"UIUC Max Fitness: {uiuc_max_fitness:.4f}, UIUC Min Fitness: {uiuc_min_fitness:.4f}")
    print(f"UIUC Max CL: {uiuc_max_cl:.4f}, UIUC Min CL: {uiuc_min_cl:.4f}")
    print(f"UIUC Max CD: {uiuc_max_cd:.4f}, UIUC Min CD: {uiuc_min_cd:.4f}")

print_uiuc_stats()
print(f'shape of uiuc_coordinates_list: {uiuc_coordinates_list[0].shape}')

# trained model paths
vae_path = "vae_epoch_200.pt"
diffusion_path_cd = "models/lucid_cd_standardized_run_1/best_model.pt"
diffusion_path_cl = "models/lucid_cl_standardized_run_1/best_model.pt"
diffusion_path_cl_thickness = "models/lucid_cl_thickness_standardized_run_1/best_model.pt"
diffusion_path_cl_cd= "models/lucid_cl_cd_standardized_run_1/best_model.pt"
diffusion_path_all = "models/lucid_all_standardized_run_1/best_model.pt"



if not os.path.exists(pkl_save_path_cd) or not os.path.exists(pkl_save_path_cl) or not os.path.exists(pkl_save_path_cl_thickness) or not os.path.exists(pkl_save_path_cl_cd) or not os.path.exists(pkl_save_path_all):
    models_to_generate = []
    if not os.path.exists(pkl_save_path_cd):
        models_to_generate.append('cd')
    if not os.path.exists(pkl_save_path_cl):
        models_to_generate.append('cl')
    if not os.path.exists(pkl_save_path_cl_thickness):
        models_to_generate.append('cl_thickness')
    if not os.path.exists(pkl_save_path_cl_cd):
        models_to_generate.append('cl_cd')
    if not os.path.exists(pkl_save_path_all):
        models_to_generate.append('all')
    for model in models_to_generate:
        if model == 'cd':
            print("Generating airfoils with CD conditioning...")
            generate_airfoils_cd(vae_path, diffusion_path_cd, pkl_save_path_cd, uiuc_cd_values, normalize=False, standardize=True)
        if model == 'cl':
            print("Generating airfoils with CL conditioning...")
            generate_airfoils_cl(vae_path, diffusion_path_cl, pkl_save_path_cl, uiuc_cl_values, normalize=False, standardize=True)
        if model == 'cl_thickness':
            print("Generating airfoils with CL and thickness conditioning...")
            generate_airfoils_cl_thickness(vae_path, diffusion_path_cl_thickness, pkl_save_path_cl_thickness, uiuc_cl_values, uiuc_max_thickness, normalize=False, standardize=True)
        if model == 'cl_cd':
            print("Generating airfoils with CL and CD conditioning...")
            generate_airfoils_cl_cd(vae_path, diffusion_path_cl_cd, pkl_save_path_cl_cd, uiuc_cl_values, uiuc_cd_values, normalize=False, standardize=True)
        if model == 'all':
            print("Generating airfoils with all conditioning values...")
            generate_airfoils_all(vae_path, diffusion_path_all, pkl_save_path_all, uiuc_cl_values, uiuc_cd_values, uiuc_max_thickness, uiuc_max_camber, normalize=False, standardize=True)
if os.path.exists(pkl_save_path_cd):
    print("Loading saved data...")
    with open(pkl_save_path_cd, 'rb') as f:
        loaded_data = pickle.load(f)
        cd_model_latent_vectors = loaded_data['latent_vectors']
        cd_model_cd_conditioning_values = loaded_data['cd_conditioning_values']
        cd_model_cd_eval_values = loaded_data['cd_eval_values']
        cd_model_cl_eval_values = loaded_data['cl_eval_values']
        cd_model_max_camber = loaded_data['maximum_camber_list']
        cd_model_max_thickness = loaded_data['maximum_thickness_list']
        cd_model_y_coords_list = loaded_data['y_coords_list']
        cd_model_gen_coefficients = loaded_data['gen_coefficients']
        cd_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cd_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cd_model_conditioning_MAE_list = loaded_data['conditioning_MAE_list']
        cd_model_airfoil_x = loaded_data['airfoil_x']
if os.path.exists(pkl_save_path_cl):
    print("Loading saved data for standardized cl model...")
    with open(pkl_save_path_cl, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_model_latent_vectors = loaded_data['latent_vectors']
        cl_model_cl_conditioning_values = loaded_data['cl_conditioning_values']
        cl_model_eval_cl_values = loaded_data['cl_eval_values']
        cl_model_eval_cd_values = loaded_data['cd_eval_values']
        cl_model_max_camber = loaded_data['maximum_camber_list']
        cl_model_max_thickness = loaded_data['maximum_thickness_list']
        cl_model_y_coords_list = loaded_data['y_coords_list']
        cl_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cl_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cl_model_conditioning_MAE_list = loaded_data['conditioning_MAE_list']
        cl_model_airfoil_x = loaded_data['airfoil_x']
if os.path.exists(pkl_save_path_cl_thickness):
    print("Loading saved data for standardized cl thickness model...")
    with open(pkl_save_path_cl_thickness, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_thickness_model_latent_vectors = loaded_data['latent_vectors']
        cl_thickness_model_cl_conditioning_values = loaded_data['cl_conditioning_values']
        cl_thickness_model_thickness_conditioning_values = loaded_data['thickness_conditioning_values']
        cl_thickness_model_eval_cl_values = loaded_data['cl_eval_values']
        cl_thickness_model_eval_cd_values = loaded_data['cd_eval_values']
        cl_thickness_model_max_thickness = loaded_data['thickness_eval_values']
        cl_thickness_model_y_coords_list = loaded_data['y_coords_list']
        cl_thickness_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_thickness_model_cl_difference_list = loaded_data['cl_difference_list']
        cl_thickness_model_thickness_difference_list = loaded_data['thickness_difference_list']
        cl_thickness_model_cl_error_list = loaded_data['cl_error_list']
        cl_thickness_model_thickness_error_list = loaded_data['thickness_error_list']
        cl_thickness_model_airfoil_x = loaded_data['airfoil_x']
if os.path.exists(pkl_save_path_cl_cd):
    print("Loading saved data...")
    with open(pkl_save_path_cl_cd, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_cd_model_latent_vectors = loaded_data['latent_vectors']
        cl_cd_model_cl_conditioning_values = loaded_data['cl_conditioning_values']
        cl_cd_model_cd_conditioning_values = loaded_data['cd_conditioning_values']
        cl_cd_model_eval_cl_values = loaded_data['cl_eval_values']
        cl_cd_model_eval_cd_values = loaded_data['cd_eval_values']
        cl_cd_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_cd_model_cl_difference_list = loaded_data['cl_difference_list']
        cl_cd_model_cd_difference_list = loaded_data['cd_difference_list']
        cl_cd_model_cl_error_list = loaded_data['cl_error_list']
        cl_cd_model_cd_error_list = loaded_data['cd_error_list']
        cl_cd_model_y_coords_list = loaded_data['y_coords_list']
        cl_cd_model_airfoil_x = loaded_data['airfoil_x']
if os.path.exists(pkl_save_path_all):
    print("Loading saved data...")
    with open(pkl_save_path_all, 'rb') as f:
        loaded_data = pickle.load(f)
        all_model_latent_vectors = loaded_data['latent_vectors']
        all_model_cl_conditioning_values = loaded_data['cl_conditioning_values']
        all_model_cd_conditioning_values = loaded_data['cd_conditioning_values']
        all_model_thickness_conditioning_values = loaded_data['thickness_conditioning_values']
        all_model_camber_conditioning_values = loaded_data['camber_conditioning_values']
        all_model_cl_eval_values = loaded_data['cl_eval_values']
        all_model_cd_eval_values = loaded_data['cd_eval_values']
        all_model_thickness_eval_values = loaded_data['thickness_eval_values']
        all_model_camber_eval_values = loaded_data['camber_eval_values']
        all_model_gen_coefficients = loaded_data['gen_coefficients']
        all_model_cl_difference_list = loaded_data['cl_difference_list']
        all_model_cd_difference_list = loaded_data['cd_difference_list']
        all_model_thickness_difference_list = loaded_data['thickness_difference_list']
        all_model_camber_difference_list = loaded_data['camber_difference_list']
        all_model_cl_error_list = loaded_data['cl_error_list']
        all_model_cd_error_list = loaded_data['cd_error_list']
        all_model_thickness_error_list = loaded_data['thickness_error_list']
        all_model_camber_error_list = loaded_data['camber_error_list']
        all_model_y_coords_list = loaded_data['y_coords_list']
        all_model_airfoil_x = loaded_data['airfoil_x']


def print_min_max_cl_cd():
    print(f'min cl cl model eval values: {min(cl_model_eval_cl_values)}')
    print(f'max cl cl model eval values: {max(cl_model_eval_cl_values)}')
    print(f'min cd cd model eval values: {min(cd_model_cd_eval_values)}')
    print(f'max cd cd model eval values: {max(cd_model_cd_eval_values)}')
print_min_max_cl_cd()


def make_boxplot_cd(cd_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cd_error_list = [value.squeeze().cpu().numpy() for value in cd_error_list]
    print(f'cd_error_liste shape: {cd_error_list[0].shape}')
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.boxplot([cd_error_list], labels=["CD"], showfliers=False)
    ax.set_title("Percent Difference in CD in Conditioning Values", fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

def make_boxplot_cd_MAE(cd_model_conditioning_MAE_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cd_model_conditioning_MAE_list = [value.squeeze().cpu().numpy() for value in cd_model_conditioning_MAE_list]
    print(f'cd_model_conditioning_MAE_list shape: {cd_model_conditioning_MAE_list[0].shape}')
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.boxplot([cd_model_conditioning_MAE_list], labels=["CD"], showfliers=False)
    ax.set_title("CD Model MAE For Evaluated Vs Conditioned Value", fontsize=title_fontsize)
    ax.set_ylabel("Mean Absolute Error", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

def make_boxplot_cl_MAE(cl_model_conditioning_MAE_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cl_model_conditioning_MAE_list = [value.squeeze().cpu().numpy() for value in cl_model_conditioning_MAE_list]
    print(f'cl_model_conditioning_MAE_list shape: {cl_model_conditioning_MAE_list[0].shape}')
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.boxplot([cl_model_conditioning_MAE_list], labels=["CL"], showfliers=False)
    ax.set_title("CL Model MAE For Evaluated Vs Conditioned Value", fontsize=title_fontsize)
    ax.set_ylabel("Mean Absolute Error", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()



def make_boxplot_cl(cl_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cl_error_list = [value.squeeze().cpu().numpy() for value in cl_error_list]
    print(f'cl_error_list shape: {cl_error_list[0].shape}')
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.boxplot([cl_error_list], labels=["CL"], showfliers=False)
    ax.set_title("Error in Conditioning Values", fontsize=title_fontsize)
    ax.set_ylabel("L1 Error", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()



def make_boxplot_combined_cd_cl(cd_error_list, cl_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cd_error_list = [value.squeeze().cpu().numpy() for value in cd_error_list]
    cl_error_list = [value.squeeze().cpu().numpy() for value in cl_error_list]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_error_list, cl_error_list], labels=["CD", "CL"], showfliers=False)
    ax.set_title("Percent Difference in Conditioning Values", fontsize=title_fontsize)
    ax.set_xlabel("Conditioning Parameter", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.show()

#make_boxplot_cd(cd_model_conditioning_error_list)
#make_boxplot_cl(cl_model_conditioning_error_list)
#make_boxplot_combined_cd_cl(cd_model_conditioning_error_list, cl_model_conditioning_error_list)
#make_boxplot_cd_MAE(cd_model_conditioning_MAE_list)
#make_boxplot_cl_MAE(cl_model_conditioning_MAE_list)

# print mean, std and quantiles for cl and cd conditioning errors
def print_stats(cd_error_list, cl_error_list):
    cd_error_list = [value.cpu().numpy() for value in cd_error_list]
    cl_error_list = [value.squeeze().cpu().numpy() for value in cl_error_list]
    cd_error_list = np.array(cd_error_list)
    cl_error_list = np.array(cl_error_list)
    cd_mean = np.mean(cd_error_list)
    cd_median = np.median(cd_error_list)
    cd_std = np.std(cd_error_list)
    cl_mean = np.mean(cl_error_list)
    cl_median = np.median(cl_error_list)
    cl_std = np.std(cl_error_list)
    cd_quantiles = np.quantile(cd_error_list, [0.25, 0.5, 0.75])
    cl_quantiles = np.quantile(cl_error_list, [0.25, 0.5, 0.75])
    print(f"CD Mean: {cd_mean:.4f}, CD Median: {cd_median:.4f} CD Std: {cd_std:.4f}, CD Quantiles: {cd_quantiles}")
    print(f"CL Mean: {cl_mean:.4f}, CL Median: {cl_median:.4f} CL Std: {cl_std:.4f}, CL Quantiles: {cl_quantiles}")

print_stats(cd_model_conditioning_error_list, cl_model_conditioning_error_list)



def make_scatter_cd(cd_conditioning_values, cd_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Convert to numpy arrays
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values]).flatten()
    cd_eval_values = np.array(cd_eval_values)

    # Linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cd_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cd_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    # Calculate metrics
    mae = mean_absolute_error(cd_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate residuals and standard deviation
    residuals = cd_eval_values - line_of_best_fit
    std_dev = np.std(residuals)

    # Plotting the line of best fit and the standard deviation band
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(cd_conditioning_values, line_of_best_fit, color='red', label=f"MAE: {mae:.4f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")

    # Shaded area for standard deviation
    ax.fill_between(cd_conditioning_values, line_of_best_fit - std_dev, line_of_best_fit + std_dev, color='orange', alpha=0.2, label='±1 Std Dev')
    ax.scatter(cd_conditioning_values, cd_eval_values, color='navy', alpha=0.5, label='Generated CD Values')

    # Labels and title
    ax.set_ylabel("Generated CD Values", fontsize=label_fontsize)
    ax.set_xlabel("CD Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    #ax.axis('equal')
    ax.set_ylim(0, max(cd_eval_values) + 0.01)
    # Set boundary thickness
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)

    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()


def make_scatter_cl(cl_conditioning_values, cl_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Convert to numpy arrays
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values]).flatten()
    cl_eval_values = np.array(cl_eval_values)

    # Linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cl_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cl_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    # Calculate R^2, slope, and intercept
    r2 = r2_score(cl_eval_values, line_of_best_fit)
    MAE = mean_absolute_error(cl_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate residuals and standard deviation
    residuals = cl_eval_values - line_of_best_fit
    std_dev = np.std(residuals)

    # Plotting the line of best fit and the standard deviation band
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(cl_conditioning_values, line_of_best_fit, color='orange', label=f"MAE: {MAE:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")

    # Shaded area for standard deviation
    ax.fill_between(cl_conditioning_values, line_of_best_fit - std_dev, line_of_best_fit + std_dev, color='green', alpha=0.2, label='±1 Std Dev')

    ax.scatter(cl_conditioning_values, cl_eval_values, color='blue', alpha=0.5, label='Generated CL Values')
    # Labels and title
    ax.set_ylabel("Generated CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    #ax.axis('equal')
    ax.set_ylim(-0.2, max(cl_eval_values) + 0.5)
    ax.set_xlim(-0.2, max(cl_conditioning_values) + 0.1)

    # Set boundary thickness
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)

    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

def make_scatter_cl_thickness(cl_conditioning_values, thickness_conditioning_values, cl_eval_values, thickness_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values])
    thickness_conditioning_values = np.array([value.cpu().numpy() for value in thickness_conditioning_values])

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cl_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cl_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cl_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cl_conditioning_values, cl_eval_values, color='blue')
    ax.plot(cl_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("CL Conditioning vs. Generated CL Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = thickness_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, thickness_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cl_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(thickness_conditioning_values, thickness_eval_values, color='blue')
    ax.plot(thickness_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("Thickness Conditioning vs. Generated Thickness", fontsize=title_fontsize)
    ax.set_ylabel("Generated CL Values", fontsize=label_fontsize)
    ax.set_xlabel("Thickness Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

def make_scatter_cl_cd(cl_conditioning_values, cd_conditioning_values, cl_eval_values, cd_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values])
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values])

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cl_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cl_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cl_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cl_conditioning_values, cl_eval_values, color='blue')
    ax.plot(cl_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("CL Conditioning vs. Generated CL Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cd_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cd_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cd_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cd_conditioning_values, cd_eval_values, color='blue')
    ax.plot(cd_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("CD Conditioning vs. Generated CD Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated CD Values", fontsize=label_fontsize)
    ax.set_xlabel("CD Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

def make_scatter_all(cl_conditioning_values, cd_conditioning_values, thickness_conditioning_values, camber_conditioning_values, cl_eval_values, cd_eval_values, thickness_eval_values, camber_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values])
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values])
    thickness_conditioning_values = np.array([value.cpu().numpy() for value in thickness_conditioning_values])
    camber_conditioning_values = np.array([value.cpu().numpy() for value in camber_conditioning_values])

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cl_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cl_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cl_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cl_conditioning_values, cl_eval_values, color='blue')
    ax.plot(cl_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("CL Conditioning vs. Generated CL Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = cd_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, cd_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(cd_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cd_conditioning_values, cd_eval_values, color='blue')
    ax.plot(cd_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("CD Conditioning vs. Generated CD Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated CD Values", fontsize=label_fontsize)
    ax.set_xlabel("CD Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = thickness_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, thickness_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)
    
    r2 = r2_score(thickness_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(thickness_conditioning_values, thickness_eval_values, color='blue')
    ax.plot(thickness_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("Thickness Conditioning vs. Generated Thickness Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated Thickness Values", fontsize=label_fontsize)
    ax.set_xlabel("Thickness Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

    # linear regression
    model = LinearRegression()
    conditioning_values_reshaped = camber_conditioning_values.reshape(-1, 1)
    model.fit(conditioning_values_reshaped, camber_eval_values)
    line_of_best_fit = model.predict(conditioning_values_reshaped)

    r2 = r2_score(camber_eval_values, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(camber_conditioning_values, camber_eval_values, color='blue')
    ax.plot(camber_conditioning_values, line_of_best_fit, color='red', label=f"R2: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
    ax.set_title("Camber Conditioning vs. Generated Camber Values", fontsize=title_fontsize)
    ax.set_ylabel("Generated Camber Values", fontsize=label_fontsize)
    ax.set_xlabel("Camber Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()

def make_combined_scatter_cd(cd_model_cd_conditioning_values, cd_model_eval_cd_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cd_values, all_model_cd_conditioning_values, all_model_cd_eval_values):
    cd_model_conditioning_values = np.array([value.cpu().numpy() for value in cd_model_cd_conditioning_values])
    cl_cd_model_conditioning_values = np.array([value.cpu().numpy() for value in cl_cd_model_cd_conditioning_values])
    all_model_conditioning_values = np.array([value.cpu().numpy() for value in all_model_cd_conditioning_values])

    # linear regression
    model = LinearRegression()
    cd_conditioning_values_reshaped = cd_model_conditioning_values.reshape(-1, 1)
    model.fit(cd_conditioning_values_reshaped, cd_model_eval_cd_values)
    cd_line_of_best_fit = model.predict(cd_conditioning_values_reshaped)

    cd_model_r2 = r2_score(cd_model_eval_cd_values, cd_line_of_best_fit)
    cd_model_slope = model.coef_[0]
    cd_model_intercept = model.intercept_

    cl_cd_model_conditioning_values_reshaped = cl_cd_model_conditioning_values.reshape(-1, 1)
    model.fit(cl_cd_model_conditioning_values_reshaped, cl_cd_model_eval_cd_values)
    cl_cd_line_of_best_fit = model.predict(cl_cd_model_conditioning_values_reshaped)

    cl_cd_model_r2 = r2_score(cl_cd_model_eval_cd_values, cl_cd_line_of_best_fit)
    cl_cd_model_slope = model.coef_[0]
    cl_cd_model_intercept = model.intercept_

    all_model_conditioning_values_reshaped = all_model_conditioning_values.reshape(-1, 1)
    model.fit(all_model_conditioning_values_reshaped, all_model_cd_eval_values)
    all_line_of_best_fit = model.predict(all_model_conditioning_values_reshaped)

    all_model_r2 = r2_score(all_model_cd_eval_values, all_line_of_best_fit)
    all_model_slope = model.coef_[0]
    all_model_intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cd_model_conditioning_values, cd_model_eval_cd_values, color='blue', label="CD Model")
    ax.plot(cd_model_conditioning_values, cd_line_of_best_fit, color='red', label=f"R2: {cd_model_r2:.2f}, Slope: {cd_model_slope:.2f}, Intercept: {cd_model_intercept:.2f}")
    ax.scatter(cl_cd_model_conditioning_values, cl_cd_model_eval_cd_values, color='green', label="CL-CD Model")
    ax.plot(cl_cd_model_conditioning_values, cl_cd_line_of_best_fit, color='purple', label=f"R2: {cl_cd_model_r2:.2f}, Slope: {cl_cd_model_slope:.2f}, Intercept: {cl_cd_model_intercept:.2f}")
    ax.scatter(all_model_conditioning_values, all_model_cd_eval_values, color='orange', label="All Model")
    ax.plot(all_model_conditioning_values, all_line_of_best_fit, color='black', label=f"R2: {all_model_r2:.2f}, Slope: {all_model_slope:.2f}, Intercept: {all_model_intercept:.2f}")
    ax.set_title("CD Conditioning vs. Generated CD Values", fontsize=22)
    ax.set_ylabel("Generated CD Values", fontsize=20)
    ax.set_xlabel("CD Conditioning Values", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.show()

def make_combined_scatter_cl(cl_model_cl_conditioning_values, cl_model_eval_cl_values, cl_cd_model_cl_conditioning_values, cl_cd_model_eval_cl_values, all_model_cl_conditioning_values, all_model_cl_eval_values):
    cl_model_conditioning_values = np.array([value.cpu().numpy() for value in cl_model_cl_conditioning_values])
    cl_cd_model_conditioning_values = np.array([value.cpu().numpy() for value in cl_cd_model_cl_conditioning_values])
    all_model_conditioning_values = np.array([value.cpu().numpy() for value in all_model_cl_conditioning_values])

    # linear regression
    model = LinearRegression()
    cl_conditioning_values_reshaped = cl_model_conditioning_values.reshape(-1, 1)
    model.fit(cl_conditioning_values_reshaped, cl_model_eval_cl_values)
    cl_line_of_best_fit = model.predict(cl_conditioning_values_reshaped)

    cl_model_r2 = r2_score(cl_model_eval_cl_values, cl_line_of_best_fit)
    cl_model_slope = model.coef_[0]
    cl_model_intercept = model.intercept_

    cl_cd_model_conditioning_values_reshaped = cl_cd_model_conditioning_values.reshape(-1, 1)
    model.fit(cl_cd_model_conditioning_values_reshaped, cl_cd_model_eval_cl_values)
    cl_cd_line_of_best_fit = model.predict(cl_cd_model_conditioning_values_reshaped)

    cl_cd_model_r2 = r2_score(cl_cd_model_eval_cl_values, cl_cd_line_of_best_fit)
    cl_cd_model_slope = model.coef_[0]
    cl_cd_model_intercept = model.intercept_

    all_model_conditioning_values_reshaped = all_model_conditioning_values.reshape(-1, 1)
    model.fit(all_model_conditioning_values_reshaped, all_model_cl_eval_values)
    all_line_of_best_fit = model.predict(all_model_conditioning_values_reshaped)

    all_model_r2 = r2_score(all_model_cl_eval_values, all_line_of_best_fit)
    all_model_slope = model.coef_[0]
    all_model_intercept = model.intercept_

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cl_model_conditioning_values, cl_model_eval_cl_values, color='blue', label="CL Model")
    ax.plot(cl_model_conditioning_values, cl_line_of_best_fit, color='red', label=f"R2: {cl_model_r2:.2f}, Slope: {cl_model_slope:.2f}, Intercept: {cl_model_intercept:.2f}")
    ax.scatter(cl_cd_model_conditioning_values, cl_cd_model_eval_cl_values, color='green', label="CL-CD Model")
    ax.plot(cl_cd_model_conditioning_values, cl_cd_line_of_best_fit, color='purple', label=f"R2: {cl_cd_model_r2:.2f}, Slope: {cl_cd_model_slope:.2f}, Intercept: {cl_cd_model_intercept:.2f}")
    ax.scatter(all_model_conditioning_values, all_model_cl_eval_values, color='orange', label="All Model")
    ax.plot(all_model_conditioning_values, all_line_of_best_fit, color='black', label=f"R2: {all_model_r2:.2f}, Slope: {all_model_slope:.2f}, Intercept: {all_model_intercept:.2f}")
    ax.set_title("CL Conditioning vs. Generated CL Values", fontsize=22)
    ax.set_ylabel("Generated CL Values", fontsize=20)
    ax.set_xlabel("CL Conditioning Values", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.show()

def make_cd_cl_scatter_and_boxplot(cd_model_cd_values, cd_model_cl_values, cl_model_cd_values, cl_model_cl_values, uiuc_cd_values, uiuc_cl_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Filter values for scatter plot
    cd_model_indices_to_plot = [i for i in range(len(cd_model_cd_values)) if cd_model_cd_values[i] < 0.07 and cd_model_cl_values[i] > 0.3]
    cl_model_indices_to_plot = [i for i in range(len(cl_model_cd_values)) if cl_model_cd_values[i] < 0.07 and cl_model_cl_values[i] > 0.3]
    
    cd_model_cd_values_to_plot = [cd_model_cd_values[i] for i in cd_model_indices_to_plot]
    cd_model_cl_values_to_plot = [cd_model_cl_values[i] for i in cd_model_indices_to_plot]
    
    cl_model_cd_values_to_plot = [cl_model_cd_values[i] for i in cl_model_indices_to_plot]
    cl_model_cl_values_to_plot = [cl_model_cl_values[i] for i in cl_model_indices_to_plot]
    
    # Calculate Lift-to-Drag ratios
    cd_model_ld_ratios = [cl/cd for cl, cd in zip(cd_model_cl_values, cd_model_cd_values)]
    cl_model_ld_ratios = [cl/cd for cl, cd in zip(cl_model_cl_values, cl_model_cd_values)]
    uiuc_ld_ratios = [cl/cd for cl, cd in zip(uiuc_cl_values, uiuc_cd_values)]
    
    # Plot Scatter Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cd_model_cd_values, cd_model_cl_values, color='blue', label="CD Model", alpha=0.5)
    ax.scatter(cl_model_cd_values, cl_model_cl_values, color='green', label="CL Model", alpha=0.5)
    ax.scatter(uiuc_cd_values, uiuc_cl_values, color='red', label="UIUC", alpha=0.25)
    ax.set_ylabel("CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CD Values", fontsize=label_fontsize)
    ax.set_xlim(0, 0.03)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()
    
    # Plot Boxplot for L/D Ratios
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_model_ld_ratios, cl_model_ld_ratios, uiuc_ld_ratios], labels=['CD Model', 'CL Model', 'UIUC'], showfliers=True)
    ax.set_ylabel("Lift-to-Drag Ratio", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

def make_cd_cl_scatter_and_plot_airfoils(cd_model_cd_values, cd_model_cl_values, cl_model_cd_values, cl_model_cl_values, uiuc_cd_values, uiuc_cl_values,
                                         uiuc_coordinates_list, cd_model_y_coords_list, cd_model_airfoil_x, cl_model_y_coords_list, cl_model_airfoil_x,
                                         title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Calculate Lift-to-Drag ratios
    cd_model_ld_ratios = np.array([cl/cd for cl, cd in zip(cd_model_cl_values, cd_model_cd_values)])
    cl_model_ld_ratios = np.array([cl/cd for cl, cd in zip(cl_model_cl_values, cl_model_cd_values)])
    uiuc_ld_ratios = np.array([cl/cd for cl, cd in zip(uiuc_cl_values, uiuc_cd_values)])

    # Find indices of the top 5 L/D ratios for each model
    top_cd_model_indices = np.argsort(cd_model_ld_ratios)[-2:]
    top_cl_model_indices = np.argsort(cl_model_ld_ratios)[-2:]
    top_uiuc_indices = np.argsort(uiuc_ld_ratios)[-2:]

    # Extract the top 5 airfoil profiles
    top_cd_model_airfoils = [(cd_model_cd_values[i], cd_model_cl_values[i], cd_model_y_coords_list[i]) for i in top_cd_model_indices]
    top_cl_model_airfoils = [(cl_model_cd_values[i], cl_model_cl_values[i], cl_model_y_coords_list[i]) for i in top_cl_model_indices]
    top_uiuc_airfoils = [(uiuc_cd_values[i], uiuc_cl_values[i], uiuc_coordinates_list[i]) for i in top_uiuc_indices]

    # Plot Scatter Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(cd_model_cd_values, cd_model_cl_values, color='blue', label="CD Model", alpha=0.5)
    ax.scatter(cl_model_cd_values, cl_model_cl_values, color='green', label="CL Model", alpha=0.5)
    ax.scatter(uiuc_cd_values, uiuc_cl_values, color='red', label="UIUC", alpha=0.25)
    ax.set_ylabel("CL Values", fontsize=label_fontsize)
    ax.set_xlabel("CD Values", fontsize=label_fontsize)
    ax.set_xlim(0, 0.03)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()
    
    # Plot Boxplot for L/D Ratios
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_model_ld_ratios, cl_model_ld_ratios, uiuc_ld_ratios], labels=['CD Model', 'CL Model', 'UIUC'], showfliers=True)
    ax.set_ylabel("Lift-to-Drag Ratio", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

    # Plot Airfoil Profiles
    fig, axs = plt.subplots(3, 2, figsize=(20, 12))
    axs = axs.flatten()

    print(f'cd model airfoil x shape: {cd_model_airfoil_x.shape}')
    print(f'cl model airfoil x shape: {cl_model_airfoil_x.shape}')
    print(f'top uiuc airfoils shape: {top_uiuc_airfoils[0][2].shape}')
    print(f'cd model top airfoils shape: {top_cd_model_airfoils[0][2].shape}')
    
    for i, (cd, cl, y_coords) in enumerate(top_cd_model_airfoils):
        print(f'y_coords shape: {y_coords.shape}')
        print(f'x_coords shape: {cd_model_airfoil_x.shape}')
        x_coords = torch.tensor(cd_model_airfoil_x)
        print(f'x_coords shape: {x_coords.shape}')
        y_coords = torch.tensor(y_coords)
        print(f'y_coords shape: {y_coords.shape}')
        axs[i].plot(x_coords[:], y_coords[:], label=f'CD Model Airfoil {i+1}', color='blue')
        axs[i].set_title(f'L/D = {cl/cd:.2f}', fontsize=title_fontsize)
    
    for i, (cd, cl, y_coords) in enumerate(top_cl_model_airfoils):
        axs[i+2].plot(cl_model_airfoil_x, y_coords, label=f'CL Model Airfoil {i+1}', color='green')
        axs[i+2].set_title(f'L/D = {cl/cd:.2f}', fontsize=title_fontsize)
    
    for i, (cd, cl, coords) in enumerate(top_uiuc_airfoils):
        print(f'coords shape: {coords.shape}')
        x_coords, y_coords = coords.squeeze()[0, :], coords.squeeze()[1, :]
        axs[i+4].plot(x_coords, y_coords, label=f'UIUC Airfoil {i+1}', color='red')
        axs[i+4].set_title(f'L/D = {cl/cd:.2f}', fontsize=title_fontsize)
    
    for ax in axs:
        ax.legend(fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=label_fontsize)
        ax.axis('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def max_camber_and_max_thickness_boxplots(cd_model_max_thickness, cl_model_max_thickness, uiuc_model_max_thickness, cd_model_max_camber, cl_model_max_camber, uiuc_model_max_camber, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Plot Boxplot for Max Thickness
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_model_max_thickness, cl_model_max_thickness, uiuc_model_max_thickness], labels=['CD Model', 'CL Model', 'UIUC'], showfliers=True)
    ax.set_ylabel("Max Thickness", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

    # Plot Boxplot for Max Camber
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_model_max_camber, cl_model_max_camber, uiuc_model_max_camber], labels=['CD Model', 'CL Model', 'UIUC'], showfliers=True)
    ax.set_ylabel("Max Camber", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

#make_scatter_cl_thickness(cl_thickness_model_cl_conditioning_values, cl_thickness_model_thickness_conditioning_values, cl_thickness_model_eval_cl_values, cl_thickness_model_max_thickness, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2)

#max_camber_and_max_thickness_boxplots(cd_model_max_thickness, cl_model_max_thickness, uiuc_max_thickness, cd_model_max_camber, cl_model_max_camber, uiuc_max_camber)

#make_cd_cl_scatter_and_plot_airfoils(cd_model_cd_eval_values, cd_model_cl_eval_values, cl_model_eval_cd_values, cl_model_eval_cl_values, uiuc_cd_values, uiuc_cl_values, uiuc_coordinates_list, cd_model_y_coords_list, cd_model_airfoil_x, cl_model_y_coords_list, cl_model_airfoil_x)

#make_cd_cl_scatter_and_boxplot(cd_model_cd_eval_values, cd_model_cl_eval_values, cl_model_eval_cd_values, cl_model_eval_cl_values, uiuc_cd_values, uiuc_cl_values)
make_scatter_cd(cd_model_cd_conditioning_values, cd_model_cd_eval_values)
make_scatter_cl(cl_model_cl_conditioning_values, cl_model_eval_cl_values)
#make_scatter_cl_cd(cl_cd_model_cl_conditioning_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cl_values, cl_cd_model_eval_cd_values)
#make_scatter_all(all_model_cl_conditioning_values, all_model_cd_conditioning_values, all_model_thickness_conditioning_values, all_model_camber_conditioning_values, all_model_cl_eval_values, all_model_cd_eval_values, all_model_thickness_eval_values, all_model_camber_eval_values)
#make_combined_scatter_cd(cd_model_cd_conditioning_values, cd_model_cd_eval_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cd_values, all_model_cd_conditioning_values, all_model_cd_eval_values)
#make_combined_scatter_cl(cl_model_cl_conditioning_values, cl_model_eval_cl_values, cl_cd_model_cl_conditioning_values, cl_cd_model_eval_cl_values, all_model_cl_conditioning_values, all_model_cl_eval_values)
#



def make_scatter_cl_cd_ratio(cl_conditioning_values, cd_conditioning_values, cl_eval_values, cd_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Convert to numpy arrays
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values]).flatten()
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values]).flatten()
    cl_eval_values = np.array(cl_eval_values)
    cd_eval_values = np.array(cd_eval_values)

    # Calculate C/D ratios for conditioning and evaluated values
    conditioning_cd_ratios = cl_conditioning_values / cd_conditioning_values
    eval_cd_ratios = cl_eval_values / cd_eval_values

    # Linear regression on the ratios
    model = LinearRegression()
    conditioning_cd_ratios_reshaped = conditioning_cd_ratios.reshape(-1, 1)
    model.fit(conditioning_cd_ratios_reshaped, eval_cd_ratios)
    line_of_best_fit = model.predict(conditioning_cd_ratios_reshaped)

    r2 = r2_score(eval_cd_ratios, line_of_best_fit)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Calculate residuals and standard deviation
    residuals = eval_cd_ratios - line_of_best_fit
    std_dev = np.std(residuals)

    # Plotting the scatter plot with line of best fit and standard deviation band
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(conditioning_cd_ratios, line_of_best_fit, color='red', label=f"R²: {r2:.2f}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")

    # Shaded area for standard deviation
    ax.fill_between(conditioning_cd_ratios, line_of_best_fit - std_dev, line_of_best_fit + std_dev, color='blue', alpha=0.2, label='±1 Std Dev')

    # Labels and title
    ax.set_title("Conditioning C/D Ratios vs. Generated C/D Ratios", fontsize=title_fontsize)
    ax.set_ylabel("Generated C/D Ratios", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning C/D Ratios", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Set boundary thickness
    for spine in ax.spines.values():
        spine.set_linewidth(boundary_thickness)

    ax.legend(fontsize=label_fontsize)
    plt.tight_layout()
    plt.show()


#make_scatter_cl_cd_ratio(cl_cd_model_cl_conditioning_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cl_values, cl_cd_model_eval_cd_values)

def make_3d_surface_cl_cd(cl_conditioning_values, cd_conditioning_values, cl_eval_values, cd_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    # Convert to numpy arrays
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values]).flatten()
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values]).flatten()
    cl_eval_values = np.array(cl_eval_values)
    cd_eval_values = np.array(cd_eval_values)

    # 3D Surface Plot for CL
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid data for surface fitting
    grid_x, grid_y = np.mgrid[min(cl_conditioning_values):max(cl_conditioning_values):100j, 
                              min(cd_conditioning_values):max(cd_conditioning_values):100j]

    grid_z_cl = griddata((cl_conditioning_values, cd_conditioning_values), cl_eval_values, (grid_x, grid_y), method='linear')

    # Plotting the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z_cl, cmap='viridis', alpha=0.8)

    # Scatter plot of the actual points
    #ax.scatter(cl_conditioning_values, cd_conditioning_values, cl_eval_values, color='red', s=50, label="Generated CL Values")

    # Labels and title
    ax.set_title("CL Conditioning and CD Conditioning vs. Generated CL Values", fontsize=title_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.set_ylabel("CD Conditioning Values", fontsize=label_fontsize)
    ax.set_zlabel("Generated CL Values", fontsize=label_fontsize)

    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # 3D Surface Plot for CD
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid data for surface fitting
    grid_z_cd = griddata((cl_conditioning_values, cd_conditioning_values), cd_eval_values, (grid_x, grid_y), method='linear')

    # Plotting the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z_cd, cmap='plasma', alpha=0.8)

    # Scatter plot of the actual points
    #ax.scatter(cl_conditioning_values, cd_conditioning_values, cd_eval_values, color='blue', s=50, label="Generated CD Values")

    # Labels and title
    ax.set_title("CL Conditioning and CD Conditioning vs. Generated CD Values", fontsize=title_fontsize)
    ax.set_xlabel("CL Conditioning Values", fontsize=label_fontsize)
    ax.set_ylabel("CD Conditioning Values", fontsize=label_fontsize)
    ax.set_zlabel("Generated CD Values", fontsize=label_fontsize)

    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

#make_3d_surface_cl_cd(cl_cd_model_cl_conditioning_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cl_values, cl_cd_model_eval_cd_values)


def condition_tsne(latent_vectors, conditioning_values, title, conditioning_label, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    latent_vectors = np.array(latent_vectors)

    tsne = TSNE(n_components=2)
    latent_vectors_embedded = tsne.fit_transform(latent_vectors)

    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    sc = axs.scatter(latent_vectors_embedded[:, 0], latent_vectors_embedded[:, 1], c=conditioning_values, cmap='viridis')

    cbar = fig.colorbar(sc, ax=axs)
    cbar.set_label(conditioning_label, fontsize=label_fontsize)
    axs.set_xlabel("TSNE 1", fontsize=label_fontsize)
    axs.set_ylabel("TSNE 2", fontsize=label_fontsize)
    axs.set_title(title, fontsize=title_fontsize)
    axs.tick_params(axis='both', labelsize=tick_fontsize)
    for spine in axs.spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

# tsne for cd model
#condition_tsne(cd_model_latent_vectors, cd_model_cd_conditioning_values, "TSNE of Latent Vectors for CD Model", "CD Conditioning Values")
#condition_tsne(cl_model_latent_vectors, cl_model_cl_conditioning_values, "TSNE of Latent Vectors for CL Model", "CL Conditioning Values")
#condition_tsne(cl_cd_model_latent_vectors, cl_cd_model_cl_conditioning_values, "TSNE of Latent Vectors for CL-CD Model", "CL Conditioning Values")
#condition_tsne(cl_cd_model_latent_vectors, cl_cd_model_cd_conditioning_values, "TSNE of Latent Vectors for CL-CD Model", "CD Conditioning Values")
#condition_tsne(all_model_latent_vectors, all_model_cl_conditioning_values, "TSNE of Latent Vectors for All Model", "CL Conditioning Values")
#condition_tsne(all_model_latent_vectors, all_model_cd_conditioning_values, "TSNE of Latent Vectors for All Model", "CD Conditioning Values")
#condition_tsne(all_model_latent_vectors, all_model_thickness_conditioning_values, "TSNE of Latent Vectors for All Model", "Thickness Conditioning Values")
#condition_tsne(all_model_latent_vectors, all_model_camber_conditioning_values, "TSNE of Latent Vectors for All Model", "Camber Conditioning Values")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def chamfer_distance(A, B):
    """
    Computes the Chamfer Distance between two sets of points A and B.
    A and B are expected to be tensors of shape [N, 1] where N is the number of points.
    """
    # Ensure the input tensors are 2D
    if A.dim() == 1:
        A = A.unsqueeze(1)  # Shape [N, 1]
    if B.dim() == 1:
        B = B.unsqueeze(1)  # Shape [M, 1]
    
    # Compute pairwise L2 distances
    pairwise_distances = torch.cdist(A, B, p=2)  # Shape [N, M]
    
    # Compute the forward Chamfer distance (A -> B)
    forward_distance = pairwise_distances.min(dim=1)[0].mean()
    
    # Compute the backward Chamfer distance (B -> A)
    backward_distance = pairwise_distances.min(dim=0)[0].mean()
    
    # Return the average of forward and backward distances
    return (forward_distance + backward_distance)/2

def calculate_pairwise_chamfer_distances(training_coordinates, generated_coordinates):
    """
    Calculates the Chamfer distances between two lists of airfoil coordinates:
    one from the training set and one from the generated samples.
    """
    chamfer_distances = []
    total_iterations = len(training_coordinates) * len(generated_coordinates)
    
    # Using tqdm for the outer loop
    with tqdm(total=total_iterations, desc="Calculating Chamfer Distances", unit="pair") as pbar:
        for train_coords in training_coordinates:
            # Move training coordinates to GPU
            train_coords_tensor = torch.tensor(train_coords, dtype=torch.float32).to(device)
            for gen_coords in generated_coordinates:
                # Move generated coordinates to GPU
                gen_coords_tensor = torch.tensor(gen_coords, dtype=torch.float32).to(device)
                # Compute Chamfer distance on GPU
                distance = chamfer_distance(train_coords_tensor, gen_coords_tensor)
                # Append the result as a CPU scalar
                chamfer_distances.append(distance.cpu().item())
                
                # Update the progress bar
                pbar.update(1)
    
    return chamfer_distances

# Prepare your data
uiuc_one_channel_coords = []
for i, coord in enumerate(uiuc_training_coordinates):
    one_channel_coordinates = torch.cat([coord.squeeze()[0,:], coord.squeeze()[1,:]])
    uiuc_one_channel_coords.append(one_channel_coordinates)

# normalize uiuc_one_channel_coords


cd_chamfer_pkl_path = "cd_model_chamfer.pkl"
cl_chamfer_pkl_path = "cl_model_chamfer.pkl"
# Calculate pairwise distances using CUDA

if not os.path.exists(cd_chamfer_pkl_path):
    with open(cd_chamfer_pkl_path, 'wb') as f:
        pairwise_distances_cd_model = calculate_pairwise_chamfer_distances(uiuc_one_channel_coords, cd_model_y_coords_list)
        pickle.dump(pairwise_distances_cd_model, f)
if not os.path.exists(cl_chamfer_pkl_path):
    with open(cl_chamfer_pkl_path, 'wb') as f:
        pairwise_distances_cl_model = calculate_pairwise_chamfer_distances(uiuc_one_channel_coords, cl_model_y_coords_list)
        pickle.dump(pairwise_distances_cl_model, f)
else:
    with open(cd_chamfer_pkl_path, 'rb') as f:
        pairwise_distances_cd_model = pickle.load(f)
    with open(cl_chamfer_pkl_path, 'rb') as f:
        pairwise_distances_cl_model = pickle.load(f)
# Plot the boxplot
def plot_chamfer_boxplot(pairwise_distances_cd_model, pairwise_distances_cl_model):
    plt.figure(figsize=(10, 5))
    plt.boxplot([pairwise_distances_cd_model, pairwise_distances_cl_model], labels=["CD Model", "CL Model"], showfliers=True)
    plt.tight_layout()
    plt.show()

plot_chamfer_boxplot(pairwise_distances_cd_model, pairwise_distances_cl_model)

# print the min and max y values for uiuc airfoils
min_y = 100
max_y = -100
print(f'uiuc_training_coords shape: {uiuc_training_coordinates[0].shape}')
for coords in uiuc_training_coordinates:
    one_channel_coords = torch.cat([coords.squeeze()[0,:], coords.squeeze()[1,:]])
    min_y = min(min_y, one_channel_coords.min())
    max_y = max(max_y, one_channel_coords.max())

# print the min and max y values for generated airfoils
min_y_cd = 100
max_y_cd = -100
for coords in cd_model_y_coords_list:
    min_y_cd = min(min_y_cd, coords.min())
    max_y_cd = max(max_y_cd, coords.max())

min_y_cl = 100
max_y_cl = -100
for coords in cl_model_y_coords_list:
    min_y_cl = min(min_y_cl, coords.min())
    max_y_cl = max(max_y_cl, coords.max())

print(f"Min Y CD: {min_y_cd}, Max Y CD: {max_y_cd}")
print(f"Min Y CL: {min_y_cl}, Max Y CL: {max_y_cl}")

print(f"Min Y UIUC: {min_y}, Max Y UIUC: {max_y}")
