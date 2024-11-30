import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from airfoil_dataset_1d import AirfoilDataset
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import splrep, BSpline
from scipy.interpolate import UnivariateSpline

def standardize_conditioning_values(conditioning_values, mean_values, std_values):
    # Standardize the conditioning values
    standardized_conditioning_values = (conditioning_values - mean_values) / std_values
    return standardized_conditioning_values

def smooth_airfoil(airfoil_lower, airfoil_upper, airfoil_x, s=0.01):
    # Separate top and bottom surfaces
    n = len(airfoil_x) // 2
    airfoil_x_lower = airfoil_x[:n]
    airfoil_x_upper = airfoil_x[n:]
    airfoil_lower = airfoil_lower[:n]
    airfoil_upper = airfoil_upper[:n]

    # Ensure x values are strictly increasing
    if not (np.all(np.diff(airfoil_x_lower) > 0) and np.all(np.diff(airfoil_x_upper) > 0)):
        print("Sorting x values to ensure they are strictly increasing")
        sorted_indices_lower = np.argsort(airfoil_x_lower)
        sorted_indices_upper = np.argsort(airfoil_x_upper)
        airfoil_x_lower_sorted = airfoil_x_lower[sorted_indices_lower]
        airfoil_lower_sorted = airfoil_lower[sorted_indices_lower]
        airfoil_x_upper_sorted = airfoil_x_upper[sorted_indices_upper]
        airfoil_upper_sorted = airfoil_upper[sorted_indices_upper]
    else:
        airfoil_x_lower_sorted = airfoil_x_lower
        airfoil_lower_sorted = airfoil_lower
        airfoil_x_upper_sorted = airfoil_x_upper
        airfoil_upper_sorted = airfoil_upper

    lower_spline = splrep(np.array(airfoil_x_lower_sorted), np.array(airfoil_lower_sorted), s=s)
    upper_spline = splrep(np.array(airfoil_x_upper_sorted), np.array(airfoil_upper_sorted), s=s)
    lower_new = BSpline(*lower_spline)(np.array(airfoil_x_lower_sorted))
    upper_new = BSpline(*upper_spline)(np.array(airfoil_x_upper_sorted))

    # Re-sort the smoothed values back to the original order
    lower_new_sorted_back = lower_new[np.argsort(sorted_indices_lower)]
    upper_new_sorted_back = upper_new[np.argsort(sorted_indices_upper)]

    
    airfoil_smooth = torch.cat([torch.tensor(upper_new_sorted_back), torch.tensor(lower_new_sorted_back)])
    return airfoil_smooth



def plot_images(airfoils, airfoil_x):
    num_airfoils = airfoils.shape[0]
    fig, axs = plt.subplots(1, num_airfoils, figsize=(num_airfoils * 5, 5))
    
    if num_airfoils == 1:
        axs = [axs]
    
    for i in range(num_airfoils):
        ax = axs[i]
        # Detach the tensor from the computation graph before converting to NumPy
        airfoil = airfoils[i]
        ax.scatter(airfoil_x, airfoil, color='black')
        ax.set_title(f'Airfoil {i+1}')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def save_images_conditional(airfoils,airfoil_x, path, conditioning, num_cols=4):
    # input tensor cl is cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device) convert to numpy
    print(f'conditioning shape: {conditioning.shape}')
    cl = conditioning[:,0].cpu().numpy()
    print(f'cl shape: {cl.shape}')
    max_camber = conditioning[:,1].cpu().numpy()
    max_thickness = conditioning[:,2].cpu().numpy()
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()

    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu()
        y_coords = torch.cat([airfoil[0], airfoil[1]])
        ax.scatter(airfoil_x, y_coords, color='black')
        cl_string = f'cl={cl[i]:.2f}'
        max_camber_string = f'max_camber={max_camber[i]:.2f}'
        max_thickness_string = f'max_thickness={max_thickness[i]:.2f}'
        ax.set_title(f'Airfoil {i+1}, {cl_string}, \n {max_camber_string},\n {max_thickness_string}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(num_airfoils, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def save_images(airfoils,airfoil_x, path, num_cols=4):
    # input tensor cl is cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device) convert to numpy
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()

    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu()
        y_coords = torch.cat([airfoil[0], airfoil[1]])
        ax.scatter(airfoil_x, y_coords, color='black')
        ax.set_title(f'Airfoil {i+1}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(num_airfoils, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def get_data(args):
    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_points_per_side, grid_size=args.grid_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)



def save_noised_airfoils(self, airfoils, airfoil_x, epoch, step, save_dir="noised_airfoils"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_step_{step}.jpg")
    self.plot_and_save_airfoils(airfoils, airfoil_x, save_path)

def plot_and_save_airfoils(self, airfoils, airfoil_x, save_path):
    num_airfoils = airfoils.shape[0]
    fig, axs = plt.subplots(1, num_airfoils, figsize=(num_airfoils * 5, 5))
    
    if num_airfoils == 1:
        axs = [axs]
    
    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu()
        y_coords = torch.cat([airfoil[0], airfoil[1]])
        ax.scatter(airfoil_x, y_coords, color='black')
        ax.set_title(f'Airfoil {i+1}')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def chamfer_distance(pred_y, target_y, x_values):
    """
    Computes the Chamfer Distance between predicted and target y-values with consistent x-values.

    :param pred_y: Predicted y-values (batch_size, num_points)
    :param target_y: Target y-values (batch_size, num_points)
    :param x_values: The fixed x-values (num_points)
    :return: Chamfer distance
    """
    x_tensor = torch.tensor(x_values).unsqueeze(0).to(pred_y.device)
    x_values = x_tensor.repeat(pred_y.shape[0], 1)
    # Combine x and y values into full coordinate pairs
    pred = torch.stack([x_values, pred_y], dim=-1)
    target = torch.stack([x_values, target_y], dim=-1)
    
    # Calculate distances
    batch_size, num_points, _ = pred.shape
    pred = pred.unsqueeze(2).repeat(1, 1, num_points, 1)
    target = target.unsqueeze(1).repeat(1, num_points, 1, 1)
    dist = torch.norm(pred - target, dim=-1)
    
    # Get minimum distances
    min_dist_pred = torch.min(dist, dim=2)[0]
    min_dist_target = torch.min(dist, dim=1)[0]
    
    # Return the Chamfer distance
    return torch.mean(min_dist_pred) + torch.mean(min_dist_target)

def get_cl_values(x, t, model, diffusion, vae, cl_values, airfoil_x):
    cl_values_list = []
    model.eval()
    denoised = diffusion.denoise(model, x, t, cl_values)
    y_values = vae.decode(denoised)
    y_values = y_values.detach().cpu().numpy()
    for y_value in y_values:
        coordinates = np.vstack([airfoil_x, y_value[0, :]]).T
        airfoil = asb.Airfoil(
            name=f'Generated Airfoil',
            coordinates=coordinates
        )
        coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
        cl = coef['CL'][0]
        cl_values_list.append(cl)
    cl_values_tensor = torch.tensor(cl_values_list).to(x.device)
    return cl_values_tensor

# function to load UIUC airfoils
def load_uiuc_airfoils(path='uiuc_airfoils.pkl'):
    if os.path.exists(path):
        print("Loading UIUC airfoils...")
        with open(path, 'rb') as f:
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
            uiuc_training_coordinates = uiuc_airfoil['train_coords_y']
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
        with open(path, 'wb') as f:
            pickle.dump(uiuc_data_to_save, f)

    # make dictionary of uiuc data
    uiuc_dict = {
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
    return uiuc_dict

