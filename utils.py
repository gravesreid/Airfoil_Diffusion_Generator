import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader
from airfoil_dataset import AirfoilDataset

def plot_images(airfoils):
    num_airfoils = airfoils.shape[0]
    fig, axs = plt.subplots(1, num_airfoils, figsize=(num_airfoils * 5, 5))
    
    if num_airfoils == 1:
        axs = [axs]
    
    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu().numpy()
        ax.scatter(airfoil[0], airfoil[1], color='black')
        ax.set_title(f'Airfoil {i+1}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def save_images(airfoils, path, cl, num_cols=4):
    # input tensor cl is cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device) convert to numpy
    cl = cl.cpu().numpy()
    num_airfoils = airfoils.shape[0]
    num_rows = (num_airfoils + num_cols - 1) // num_cols  # Ensure we cover all airfoils
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    
    axs = axs.flatten()

    for i in range(num_airfoils):
        ax = axs[i]
        airfoil = airfoils[i].cpu().numpy()
        print(f'Airfoil: {airfoil}')
        ax.scatter(airfoil[0], airfoil[1], color='black')
        cl_string = f'cl={cl[i][0]:.2f}'
        ax.set_title(f'Airfoil {i+1}, {cl_string}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

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

