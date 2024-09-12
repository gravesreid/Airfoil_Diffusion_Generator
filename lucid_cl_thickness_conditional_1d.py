import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils_1d_2channel import * 
import logging
from airfoil_dataset_1d_2channel import AirfoilDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import math
import matplotlib.pyplot as plt
import aerosandbox as asb
from LucidDiffusion import *
import pickle

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# load uiuc airfoil data
uiuc_path = '/home/reid/Projects/Airfoil_Diffusion/conditional_airfoil_diffusion/uiuc_airfoils.pkl'

with open(uiuc_path, 'rb') as f:
    uiuc_data = pickle.load(f)
    uiuc_cl_mean = uiuc_data['uiuc_cl_mean']
    uiuc_cl_std = uiuc_data['uiuc_cl_std']
    uiuc_cd_mean = uiuc_data['uiuc_cd_mean']
    uiuc_cd_std = uiuc_data['uiuc_cd_std']
    uiuc_max_cl = uiuc_data['uiuc_max_cl']
    uiuc_max_cd = uiuc_data['uiuc_max_cd']
    uiuc_min_cl = uiuc_data['uiuc_min_cl']
    uiuc_min_cd = uiuc_data['uiuc_min_cd']
    uiuc_max_thickness_mean = uiuc_data['uiuc_max_thickness_mean']
    uiuc_max_thickness_std = uiuc_data['uiuc_max_thickness_std']


def save_images_conditional(airfoils,airfoil_x, path, conditioning, num_cols=4):
    # input tensor cl is cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device) convert to numpy
    cl = conditioning[:,0].cpu().numpy()
    cd = conditioning[:,1].cpu().numpy()
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
        cd_string = f'cd={cd[i]:.2f}'
        ax.set_title(f'Airfoil {i+1}, {cl_string}, {cd_string}')
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide any remaining empty subplots
    for j in range(num_airfoils, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def train(args):
    setup_logging(args.run_name)
    device = args.device

    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Unet1DConditional(12, cond_dim=2, dim_mults=(1, 2, 4), channels=2, dropout=0.).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10, verbose=True)
    l1 = nn.L1Loss()
    diffusion = GaussianDiffusion1D(model, seq_length=args.num_airfoil_points, objective='pred_noise', timesteps=1000).to(device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    # Wandb setup
    wandb.init(project='conditional_latent_airfoil_diffusion', name=args.run_name, config=args)
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    airfoil_x = dataset.get_x()

    training_loss = []

    best_loss = float('inf')
    patience = 200
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, airfoil in enumerate(pbar):
            train_coords = airfoil['train_coords_y'].to(device).float()
            cl = airfoil['CL'].to(device).float().unsqueeze(1)
            max_thickness = airfoil['max_thickness'].to(device).float().unsqueeze(1)
            # normalize cl and cd
            #cl = (cl - uiuc_min_cl) / (uiuc_max_cl - uiuc_min_cl)
            #cd = (cd - uiuc_min_cd) / (uiuc_max_cd - uiuc_min_cd)
            # standardize cl and cd
            cl = (cl - uiuc_cl_mean) / uiuc_cl_std
            #cd = (cd - uiuc_cd_mean) / uiuc_cd_std
            max_thickness = (max_thickness - uiuc_max_thickness_mean) / uiuc_max_thickness_std
            # shift mean away from 0
            cl = cl 
            conditioning = torch.cat([cl, max_thickness], dim=1)

            # Pass train coords through VAE
            t = torch.randint(0, diffusion.num_timesteps, (train_coords.shape[0],), device=device).long()
            noise = torch.randn_like(train_coords, device=device)
            x_t = diffusion.q_sample(train_coords, t, noise=noise)

            if torch.rand(1).item() < .2:
                cl = None

            predicted_noise = diffusion.model(x_t, t, conditioning=conditioning)
            cl_error = l1(noise, predicted_noise)

            # Optionally calculate CL error if CL is provided
            '''
            if cl is not None:
                gen_cl = get_cl_values(x_t, t, model, diffusion, vae, cl, airfoil_x).to(device)
                cl_error += l1(cl.squeeze(1), gen_cl)
            '''

            optimizer.zero_grad()
            cl_error.backward()
            optimizer.step()

            epoch_loss += cl_error.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=cl_error.item(), learning_rate=current_lr)
            wandb.log({
                "Batch Loss": cl_error.item(),
                "Learning Rate": current_lr,
                "Epoch": epoch
            })
            logger.add_scalar(f"loss: {epoch}", cl_error.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join("models", args.run_name, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epoch} epochs.")
            break

        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}, epochs no improvement: {epochs_no_improve}, best loss: {best_loss}")

        if epoch % 100 == 0:
            cl = torch.linspace(0, 1, 5).unsqueeze(1).to(device)
            max_thickness = torch.linspace(-1, 1, 5).unsqueeze(1).to(device)
            combined = torch.cat([cl, max_thickness], dim=1)
            sampled_images = diffusion.sample(batch_size=5, conditioning=combined)
            save_images_conditional(sampled_images, airfoil_x, os.path.join("results", args.run_name, f"{epoch}.jpg"), combined)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            wandb.log({"Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}.jpg"))],
                       "learning_rate": current_lr})

    # Save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    loss_curve_path = os.path.join("results", args.run_name, "training_loss.jpg")
    plt.savefig(loss_curve_path)

    # Log the loss curve image to WandB
    wandb.log({"Training Loss Curve": wandb.Image(loss_curve_path)})
    wandb.save(loss_curve_path)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_cl_thickness_standardized_run_1")
    parser.add_argument('--epochs', type=int, default=5001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()