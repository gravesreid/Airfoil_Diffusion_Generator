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
pkl_save_path_cd = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cd_normalized.pkl'
pkl_save_path_cl = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cl_normalized.pkl'
pkl_save_path_cl_cd = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cl_cd_normalized.pkl'
pkl_save_path_all = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_all_standardized.pkl'
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
diffusion_path_cd = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cd_normalized_run_1/best_model.pt"
diffusion_path_cl = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_normalized_run_1/best_model.pt"
diffusion_path_cl_cd= "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_cd_normalized_run_1/best_model.pt"
diffusion_path_all = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_all_standardized_run_1/best_model.pt"



if os.path.exists(pkl_save_path_cd):
    print("Loading saved data...")
    with open(pkl_save_path_cd, 'rb') as f:
        loaded_data = pickle.load(f)
        cd_model_latent_vectors = loaded_data['latent_vectors']
        cd_model_cd_conditioning_values = loaded_data['cd_conditioning_values']
        cd_model_cd_eval_values = loaded_data['cd_eval_values']
        cd_model_y_coords_list = loaded_data['y_coords_list']
        cd_model_gen_coefficients = loaded_data['gen_coefficients']
        cd_model_gen_max_camber = loaded_data['gen_max_camber']
        cd_model_gen_max_thickness = loaded_data['gen_max_thickness']
        cd_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cd_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cd_model_airfoil_x = loaded_data['airfoil_x']
if os.path.exists(pkl_save_path_cl):
    print("Loading saved data...")
    with open(pkl_save_path_cl, 'rb') as f:
        loaded_data = pickle.load(f)
        cl_model_latent_vectors = loaded_data['latent_vectors']
        cl_model_cl_conditioning_values = loaded_data['cl_conditioning_values']
        cl_model_eval_cl_values = loaded_data['cl_eval_values']
        cl_model_y_coords_list = loaded_data['y_coords_list']
        cl_model_gen_coefficients = loaded_data['gen_coefficients']
        cl_model_conditioning_error_list = loaded_data['conditioning_error_list']
        cl_model_conditioning_difference_list = loaded_data['conditioning_difference_list']
        cl_model_airfoil_x = loaded_data['airfoil_x']
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
            generate_airfoils_cd(vae_path, diffusion_path_cd, pkl_save_path_cd, uiuc_cd_values, normalize=True, standardize=False)
        elif model == 'cl':
            print("Generating airfoils with CL conditioning...")
            generate_airfoils_cl(vae_path, diffusion_path_cl, pkl_save_path_cl, uiuc_cl_values, normalize=True, standardize=False)
        elif model == 'cl_cd':
            print("Generating airfoils with CL and CD conditioning...")
            generate_airfoils_cl_cd(vae_path, diffusion_path_cl_cd, pkl_save_path_cl_cd, uiuc_cl_values, uiuc_cd_values, normalize=True, standardize=False)
        elif model == 'all':
            print("Generating airfoils with all conditioning values...")
            generate_airfoils_all(vae_path, diffusion_path_all, pkl_save_path_all, uiuc_cl_values, uiuc_cd_values, uiuc_max_thickness, uiuc_max_camber, normalize=True, standardize=False)

def make_boxplot_cd(cd_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cd_error_list = [value.cpu().numpy() for value in cd_error_list]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cd_error_list], labels=["CD"], showfliers=False)
    ax.set_title("Error in Conditioning Values", fontsize=title_fontsize)
    ax.set_ylabel("L1 Error", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

def make_boxplot_cl(cl_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    cl_error_list = [value.cpu().numpy() for value in cl_error_list]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cl_error_list], labels=["CL"], showfliers=False)
    ax.set_title("Error in Conditioning Values", fontsize=title_fontsize)
    ax.set_ylabel("L1 Error", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

def make_boxplot_cl_cd(cl_error_list, cd_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cl_error_list, cd_error_list], labels=["CL", "CD"], showfliers=False)
    ax.set_title("Error in Conditioning Values", fontsize=title_fontsize)
    ax.set_ylabel("L1 Error", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

def make_boxplot_all(cl_error_list, cd_error_list, thickness_error_list, camber_error_list, title_fontsize=22, label_fontsize=20, tick_fontsize=18):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.boxplot([cl_error_list, cd_error_list, thickness_error_list, camber_error_list], labels=["CL", "CD", "Thickness", "Camber"], showfliers=False)
    ax.set_title("Error in Conditioning Values", fontsize=title_fontsize)
    ax.set_ylabel("L1 Error", fontsize=label_fontsize)
    ax.set_xlabel("Conditioning Values", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    plt.show()

make_boxplot_cd(cd_model_conditioning_error_list)
make_boxplot_cl(cl_model_conditioning_error_list)
#make_boxplot_cl_cd(cl_cd_model_cl_error_list, cl_cd_model_cd_error_list)
#make_boxplot_all(all_model_cl_error_list, all_model_cd_error_list, all_model_thickness_error_list, all_model_camber_error_list)


def make_scatter_cd(cd_conditioning_values, cd_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    cd_conditioning_values = np.array([value.cpu().numpy() for value in cd_conditioning_values])

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

def make_scatter_cl(cl_conditioning_values, cl_eval_values, title_fontsize=22, label_fontsize=20, tick_fontsize=18, boundary_thickness=2):
    cl_conditioning_values = np.array([value.cpu().numpy() for value in cl_conditioning_values])

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

make_scatter_cd(cd_model_cd_conditioning_values, cd_model_cd_eval_values)
make_scatter_cl(cl_model_cl_conditioning_values, cl_model_eval_cl_values)
make_scatter_cl_cd(cl_cd_model_cl_conditioning_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cl_values, cl_cd_model_eval_cd_values)
make_scatter_all(all_model_cl_conditioning_values, all_model_cd_conditioning_values, all_model_thickness_conditioning_values, all_model_camber_conditioning_values, all_model_cl_eval_values, all_model_cd_eval_values, all_model_thickness_eval_values, all_model_camber_eval_values)
make_combined_scatter_cd(cd_model_cd_conditioning_values, cd_model_cd_eval_values, cl_cd_model_cd_conditioning_values, cl_cd_model_eval_cd_values, all_model_cd_conditioning_values, all_model_cd_eval_values)
make_combined_scatter_cl(cl_model_cl_conditioning_values, cl_model_eval_cl_values, cl_cd_model_cl_conditioning_values, cl_cd_model_eval_cl_values, all_model_cl_conditioning_values, all_model_cl_eval_values)

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
condition_tsne(cd_model_latent_vectors, cd_model_cd_conditioning_values, "TSNE of Latent Vectors for CD Model", "CD Conditioning Values")
condition_tsne(cl_model_latent_vectors, cl_model_cl_conditioning_values, "TSNE of Latent Vectors for CL Model", "CL Conditioning Values")
condition_tsne(cl_cd_model_latent_vectors, cl_cd_model_cl_conditioning_values, "TSNE of Latent Vectors for CL-CD Model", "CL Conditioning Values")
condition_tsne(cl_cd_model_latent_vectors, cl_cd_model_cd_conditioning_values, "TSNE of Latent Vectors for CL-CD Model", "CD Conditioning Values")
condition_tsne(all_model_latent_vectors, all_model_cl_conditioning_values, "TSNE of Latent Vectors for All Model", "CL Conditioning Values")
condition_tsne(all_model_latent_vectors, all_model_cd_conditioning_values, "TSNE of Latent Vectors for All Model", "CD Conditioning Values")
condition_tsne(all_model_latent_vectors, all_model_thickness_conditioning_values, "TSNE of Latent Vectors for All Model", "Thickness Conditioning Values")
condition_tsne(all_model_latent_vectors, all_model_camber_conditioning_values, "TSNE of Latent Vectors for All Model", "Camber Conditioning Values")