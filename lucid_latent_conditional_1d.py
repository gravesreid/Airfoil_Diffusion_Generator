import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils_1d import * 
import logging
from airfoil_dataset_1d import AirfoilDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import math
import matplotlib.pyplot as plt
from vae import Encoder, Decoder, VAE
import aerosandbox as asb
from LucidDiffusion import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device

    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Unet1DConditional(8, cond_dim=1, channels=1).to(device)
    vae = VAE(args.num_airfoil_points * 2, args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_model_path, weights_only=True))
    vae.eval()
    vae.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True)
    l1 = nn.L1Loss()
    diffusion = GaussianDiffusion1D(model, seq_length=args.latent_dim, objective='pred_noise', timesteps=1000).to(device)

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

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, airfoil in enumerate(pbar):
            train_coords = airfoil['train_coords_y'].to(device).float().unsqueeze(1)
            cl = airfoil['CL'].to(device).float().unsqueeze(1)

            # Pass train coords through VAE
            mu, _ = vae.enc(train_coords)
            mu.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (mu.shape[0],), device=device).long()
            noise = torch.randn_like(mu, device=device)
            x_t = diffusion.q_sample(mu, t, noise=noise)

            if torch.rand(1).item() < .1:
                cl = None

            predicted_noise = diffusion.model(x_t, t, cl)
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
            logger.add_scalar(f"loss: {epoch}", cl_error.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}")

        if epoch % 100 == 0:
            cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device)
            sampled_images = diffusion.sample(batch_size=5, conditioning=cl)
            sampled_images = vae.decode(sampled_images)
            save_images_conditional(sampled_images, airfoil_x, os.path.join("results", args.run_name, f"{epoch}.jpg"), cl)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            wandb.log({"Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}.jpg"))],
                       "learning_rate": current_lr})

    # Save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join("results", args.run_name, "training_loss.jpg"))
    wandb.save(os.path.join("results", args.run_name, "training_loss.jpg"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_latent_run_3")
    parser.add_argument('--epochs', type=int, default=1001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--vae_model_path', type=str, default='vae_epoch_200.pt')
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()