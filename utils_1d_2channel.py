import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from airfoil_dataset_1d import AirfoilDataset
import numpy as np

def smooth_airfoil(airfoil, airfoil_x, method = 'moving average', window = 3):
    if method == 'moving average':
        airfoil_smooth = np.convolve(airfoil, np.ones(window)/window, mode='same')
    elif method == 'spline':
        from scipy.interpolate import interp1d
        f = interp1d(airfoil_x, airfoil, kind='cubic')
        airfoil_smooth = f(airfoil_x)
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
    cl = conditioning[:,0].cpu().numpy()
    cd = conditioning[:,1].cpu().numpy()
    max_camber = conditioning[:,2].cpu().numpy()
    max_thickness = conditioning[:,3].cpu().numpy()
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()

    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu()
        y_coords = torch.cat([airfoil[0], airfoil[1]])
        ax.scatter(airfoil_x, y_coords, color='black')
        cl_string = f'cl={cl[i][0]:.2f}'
        cd_string = f'cd={cd[i][0]:.2f}'
        max_camber_string = f'max_camber={max_camber[i][0]:.2f}'
        max_thickness_string = f'max_thickness={max_thickness[i][0]:.2f}'
        ax.set_title(f'Airfoil {i+1}, {cl_string}, {cd_string}, {max_camber_string}, {max_thickness_string}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(num_airfoils, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def save_images(airfoils,airfoil_x, path, num_cols=4):
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()


    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu().numpy()
        ax.scatter(airfoil_x, airfoil[0,:], color='black')
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