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
from sklearn.decomposition import PCA
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pkl_save_path = 'gen_airfoils_cl.pkl'

uiuc_pkl_path = 'uiuc_airfoils.pkl'
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
    uiuc_airfoil_path = 'coord_seligFmt'
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
        cl = uiuc_airfoil['CD'][0]
        uiuc_cl_values.append(cl)
        uiuc_cd_values.append(cl)
        max_camber = uiuc_airfoil['max_camber']
        uiuc_max_camber.append(max_camber)
        max_thickness = uiuc_airfoil['max_thickness']
        uiuc_max_thickness.append(max_thickness)
        name = uiuc_airfoil['name']
        uiuc_names.append(name)
        fitness = cl/cl
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

vae_path = "vae_epoch_200.pt"
diffusion_path = "models/lucid_cl_standardized_run_1/best_model.pt"





if os.path.exists(pkl_save_path):
    print("Loading saved data...")
    with open(pkl_save_path, 'rb') as f:
        loaded_data = pickle.load(f)
        #cl_values = loaded_data['cl_values']
        y_coords_list = loaded_data['y_coords_list']
        gen_coefficients = loaded_data['gen_coefficients']
        gen_max_camber = loaded_data['gen_max_camber']
        gen_max_thickness = loaded_data['gen_max_thickness']
        conditioning_error_list = loaded_data['conditioning_error_list']
        conditioning_difference_list = loaded_data['conditioning_difference_list']
        airfoil_x = loaded_data['airfoil_x']
else:
    # Parameters
    airfoil_dim = 200
    n_samples = 6
    unet_dim = 12
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Load the trained diffusion model
    diffusion_model = Unet1DConditional(unet_dim, cond_dim=1, channels=2, dim_mults=(1,2,4)).to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, weights_only=True))
    diffusion_model.eval()
    diffusion = GaussianDiffusion1D(diffusion_model, seq_length=int(airfoil_dim/2)).to(device)

    # Initialize dataset
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    airfoil_x = dataset.get_x()

    # Generate latent vectors with diffusion
    # make conditioning tensor of shape (n_samples, 1)
    # Define ranges for CL and CD
    # get range of cd values from uiuc airfoils
    min_cl = min(uiuc_cl_values)
    print(f'min_cd: {min_cl}')
    max_cl = max(uiuc_cl_values)
    print(f'max_cd: {max_cl}')
    cl_range = torch.linspace(min_cl, max_cl, 20).unsqueeze(1).to(device)

    # Create a grid of CL and CD values
    n_samples = cl_range.shape[0]

    # Generate airfoils for each CL, CD pair
    cl_values = []

    samples_generated = 0
    batch_size = 100  # Set the batch size

    y_coords_list = []
    airfoil_list = []
    gen_coefficients = []
    gen_max_camber = []
    gen_max_thickness = []
    conditioning_error_list = []
    conditioning_difference_list = []
    for conditioning_values in cl_range:
        conditioning_values = conditioning_values.repeat(batch_size, 1)
        generated_y = diffusion.sample(batch_size=batch_size, conditioning=conditioning_values)
        
        for i in range(batch_size):
            # get y_coords into one channel
            y_coords = torch.cat([generated_y[i, 0, :], generated_y[i, 1, :]])
            
            
            # Store the conditioning values for each sample
            cl_values.append(conditioning_values[0].detach().cpu())

            # store y_coords for plotting andCreate an airfoil object
            y_coords = y_coords.detach().cpu().numpy()
            y_coords_list.append(y_coords)
            coordinates = np.vstack([airfoil_x, y_coords]).T
            airfoil = asb.Airfoil(
                name = f'Generated Airfoil {i+1}',
                coordinates = coordinates
            )
            coefficients = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
            cl = coefficients['CL'][0]
            conditioning_difference = cl - conditioning_values[0]
            conditioning_difference_list.append(conditioning_difference)
            conditioning_error = torch.abs(cl - conditioning_values[0])/torch.abs(conditioning_values[0])
            conditioning_error_list.append(conditioning_error)
            gen_coefficients.append(coefficients)
            max_camber = airfoil.max_camber()
            gen_max_camber.append(max_camber)
            max_thickness = airfoil.max_thickness()
            gen_max_thickness.append(max_thickness)

        samples_generated += batch_size
        print(f"Generated airfoil batch: {samples_generated}/{n_samples * batch_size}")

    data_to_save = {
        'cl_values': cl_values,
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




# Perform PCA on the generated airfoils
airfoil_matrix = np.array(y_coords_list)
print(f'airfoil_matrix shape: {airfoil_matrix.shape}')
pca = PCA(n_components=2, whiten=True, random_state=42, svd_solver='full', power_iteration_normalizer='LU')
pca_result = pca.fit_transform(airfoil_matrix)

# Create subplots for PCA results
fig, axs = plt.subplots(1, 1, figsize=(8, 6))



# Perform PCA on the UIUC airfoils
uiuc_airfoil_matrix = np.array([coords.flatten() for coords in uiuc_coordinates_list])
print(f'uiuc_airfoil_matrix shape: {uiuc_airfoil_matrix.shape}')
pca_2 = PCA(n_components=2, whiten=True, random_state=42, svd_solver='full', power_iteration_normalizer='LU')
uiuc_pca_result = pca_2.fit_transform(uiuc_airfoil_matrix)

# Plot the PCA results for both generated and UIUC airfoils
sc2 = axs.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', marker='o', label='Generated Airfoils', alpha=0.5)
axs.scatter(uiuc_pca_result[:, 0], uiuc_pca_result[:, 1], c='red', marker='o', label='UIUC Airfoils', alpha=0.5)
axs.set_xlabel('Principal Component 1')
axs.set_ylabel('Principal Component 2')
axs.set_title('PCA Comparison of Generated and UIUC Airfoils')
axs.legend()

plt.show()