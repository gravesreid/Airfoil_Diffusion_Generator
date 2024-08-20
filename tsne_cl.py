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

pkl_save_path = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/gen_airfoils_cl.pkl'

uiuc_pkl_path = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/uiuc_airfoils.pkl'
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
    uiuc_data_to_save = {
        'uiuc_coordinates': uiuc_coordinates_list,
        'uiuc_cl_values': uiuc_cl_values,
        'uiuc_cd_values': uiuc_cd_values,
        'uiuc_max_camber': uiuc_max_camber,
        'uiuc_max_thickness': uiuc_max_thickness,
        'uiuc_names': uiuc_names
    }
    with open(uiuc_pkl_path, 'wb') as f:
        pickle.dump(uiuc_data_to_save, f)


vae_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/vae_epoch_200.pt"
diffusion_path = "/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/models/lucid_cl_run_1/best_model.pt"

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


if os.path.exists(pkl_save_path):
    print("Loading saved data...")
    with open(pkl_save_path, 'rb') as f:
        loaded_data = pickle.load(f)
        latent_vectors = loaded_data['latent_vectors']
        cl_values = loaded_data['cl_values']
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
    cl_range = torch.linspace(-0.2, 2, 100).unsqueeze(1).to(device)

    # Create a grid of CL and CD values
    n_samples = cl_range.shape[0]

    # Generate airfoils for each CL, CD pair
    latent_vectors = []
    cl_values = []

    samples_generated = 0
    batch_size = 3  # Set the batch size

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
            
            # Get the latent vector using the VAE encoder
            latent_vector = vae_model.enc(y_coords)
            latent_vector = latent_vector[0].detach().cpu().numpy()
            latent_vector = torch.tensor(latent_vector)
            latent_vectors.append(latent_vector)
            
            # Store the conditioning values for each sample
            cl_values.append(conditioning_values[0].detach().cpu())

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
        'latent_vectors': latent_vectors,
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

def make_hist_and_boxplot(error_list, title_fontsize=22, label_fontsize=20, boundary_thickness=2):
    for i, error in enumerate(error_list):
        error_list[i] = error.cpu().numpy()
    error_array = np.array(error_list)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].hist(error_array, bins=20, alpha=0.5)
    axs[0].set_title('CL Difference Histogram', fontsize=title_fontsize)
    axs[0].set_xlabel('Percent Difference', fontsize=label_fontsize)
    axs[0].set_ylabel('Frequency', fontsize=label_fontsize)
    axs[0].tick_params(axis='both', labelsize=16)
    for spine in axs[0].spines.values():
        spine.set_linewidth(boundary_thickness)
    axs[1].boxplot(error_array, vert=False)
    axs[1].set_title('CL Difference Boxplot', fontsize=title_fontsize)
    axs[1].set_xlabel('Percent Difference', fontsize=label_fontsize)
    axs[1].tick_params(axis='x', labelsize=16)
    for spine in axs[1].spines.values():
        spine.set_linewidth(boundary_thickness)
    plt.tight_layout()
    plt.show()

# Call the plotting function with the calculated errors
make_hist_and_boxplot(conditioning_error_list, title_fontsize=22, label_fontsize=20, boundary_thickness=2)


# function to make scatter plot of input vs output conditioning values, with line of best fit

def plot_conditioning_scatter(input_values, output_values, title_fontsize=22, label_fontsize=20, boundary_thickness=2):
    input_values = np.array([value.cpu().numpy() for value in input_values])

    # Perform linear regression
    model = LinearRegression()
    input_values_reshaped = input_values.reshape(-1, 1)
    model.fit(input_values_reshaped, output_values)
    line_of_best_fit = model.predict(input_values_reshaped)
    
    # Calculate slope, intercept, and R²
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(output_values, line_of_best_fit)
    
    # Create the scatter plot
    fig, axs = plt.subplots(figsize=(10, 6))
    axs.scatter(input_values, output_values, label='Data Points')
    
    # Plot the line of best fit
    axs.plot(input_values, line_of_best_fit, color='red', label=f'Best Fit Line: y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}')
    
    # Plot the y = x reference line
    axs.plot(input_values, input_values, color='green', linestyle='--', label='y = x (One to one match)')
    
    # Set title and labels
    axs.set_title('Input vs Output Conditioning Values', fontsize=title_fontsize)
    axs.set_xlabel('Input CL Value', fontsize=label_fontsize)
    axs.set_ylabel('Output CL Value', fontsize=label_fontsize)
    
    # Customize ticks and spine thickness
    axs.tick_params(axis='both', labelsize=16)
    for spine in axs.spines.values():
        spine.set_linewidth(boundary_thickness)
    
    # Add legend
    axs.legend(fontsize=16)
    
    plt.tight_layout()
    plt.show()

# Call the plotting function with the calculated errors
gen_cl_values = [coef['CL'][0] for coef in gen_coefficients]
plot_conditioning_scatter(cl_values, gen_cl_values, title_fontsize=22, label_fontsize=20, boundary_thickness=2)

# function to plot airfoil contours for generated and uiuc airfoils, with range of cl values
def plot_uiuc_and_gen_contours(uiuc_coordinates, gen_y_coords_list, uiuc_cl_values, gen_cl_values, uiuc_names, uiuc_max_thickness, uiuc_max_camber, gen_max_thickness, gen_max_camber, airfoil_x, cl_range = np.arange(-.2, 1.2, 5), title_fontsize=22):
    # select airfoils for each cl value from gen and uiuc airfoils
    for i, cl in enumerate(cl_range):
        # get closest cl value from gen airfoils
        gen_cl_values = np.array(gen_cl_values)
        gen_cl_diff = np.abs(gen_cl_values - cl)
        gen_cl_index = np.argmin(gen_cl_diff)
        gen_y_coords = gen_y_coords_list[gen_cl_index]
        gen_y_coords = smooth_y_coords(gen_y_coords, method='moving_average', window_size=3)
        
        # get closest cl value from uiuc airfoils
        uiuc_cl_values = np.array(uiuc_cl_values)
        uiuc_cl_diff = np.abs(uiuc_cl_values - cl)
        uiuc_cl_index = np.argmin(uiuc_cl_diff)
        uiuc_coords = uiuc_coordinates[uiuc_cl_index].squeeze()
        uiuc_names = np.array(uiuc_names)
        uiuc_max_thickness = np.array(uiuc_max_thickness)
        uiuc_max_camber = np.array(uiuc_max_camber)
        print(f'uiuc_coords shape: {uiuc_coords.shape}')
        
        # plot the airfoil contours
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        axs[0].plot(uiuc_coords[0,:], uiuc_coords[1,:], color='black')
        axs[0].set_title(
            f'UIUC {uiuc_names[uiuc_cl_index].item()},\n'
            f'CL: {uiuc_cl_values[uiuc_cl_index].item():.4f} \n',
            fontsize=title_fontsize
        )
        axs[0].set_aspect('equal')
        axs[0].axis('off')
        
        axs[1].plot(airfoil_x, gen_y_coords, color='black')
        axs[1].set_title(
            f'Generated Airfoil, \n'
            f'CL: {gen_cl_values[gen_cl_index]:.4f} \n'
            , fontsize=title_fontsize
                )
        axs[1].set_aspect('equal')
        axs[1].axis('off')
        
        plt.show()

plot_uiuc_and_gen_contours(uiuc_coordinates_list, y_coords_list, uiuc_cl_values, gen_cl_values, uiuc_names, uiuc_max_thickness, uiuc_max_camber, gen_max_thickness, gen_max_camber, airfoil_x, cl_range=np.linspace(-0.2, 2, 5), title_fontsize=22)

# t-SNE stuff
latent_vectors = np.array(latent_vectors)

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2)
latent_tsne = tsne.fit_transform(latent_vectors)
print(f'Latent t-SNE shape: {latent_tsne.shape}')
print(f'CL values shape: {len(cl_values)}')
print(f'CL values device: {cl_values[0].device}')

# Plot t-SNE with colors based on conditioning values
fig, axs = plt.subplots(figsize=(20, 10))
sc = axs.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=cl_values, cmap='viridis', label='Generated Airfoils', alpha=0.5)

# Add colorbar using the figure object
cbar = fig.colorbar(sc, ax=axs)
cbar.set_label('CL Conditioning Value', fontsize=24)
# only show latent points
axs.set_xlabel('t-SNE Dimension 1', fontsize=24)
axs.set_ylabel('t-SNE Dimension 2', fontsize=24)
axs.tick_params(axis='both', labelsize=20)
axs.set_xbound(-100, 100)
axs.set_ybound(-100, 100)
plt.show()

