import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils_1d import * 
from modules import UNet_conditional, EMA, MPP
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

class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-7, beta_end=5e-5, num_airfoil_points=200, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        #self.beta = self._cosine_variance_schedule(noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumsum(torch.log(self.alpha), dim=0).exp()

        self.num_airfoil_points = num_airfoil_points

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 4e-5):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clamp(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    
    def _diffusion_step(self, x, t, predicted_noise, alpha, alpha_hat, beta, noise):
        """
        Perform a single step of the diffusion process.
        """
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        return x

    def noise_images(self, x, t, visualize=False):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def denoise(self, model, x, t, conditioning):
        model.eval()
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x

    def sample(self, model, n, conditioning, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.num_airfoil_points)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
    
    

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    #model = UNet_conditional(c_in=1, c_out=1, cond_dim=args.cond_dim, time_dim=64, base_dim=8).to(device)
    model = Unet1D(args.latent_dim, channels=1).to(device)
    vae = VAE(args.num_airfoil_points*2, args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_model_path))
    vae.eval()
    vae.requires_grad_(False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True)
    mse = nn.MSELoss(reduction='none')
    l1 = nn.L1Loss()
    #diffusion = Diffusion(num_airfoil_points=args.latent_dim, device=device)
    diffusion = GaussianDiffusion1D(model, seq_length=args.latent_dim)
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
            cd = airfoil['CD'].to(device)
            #pass train coords through vae
            mu, _ = vae.enc(train_coords)
            t = diffusion.sample_timesteps(mu.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(mu, t)
            if np.random.random() < 0.1:
                cl = None
            predicted_noise = model(x_t, t, cl)
            #loss = mse(noise, predicted_noise)
            if cl is not None:
                gen_cl = get_cl_values(x_t,t,model,diffusion,vae,cl,airfoil_x).to(device)
                cl_error = l1(cl.squeeze(1), gen_cl)
            else:
                cl_error = 0
            loss = l1(noise, predicted_noise) + cl_error

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=loss.item(), learning_rate=current_lr)
            logger.add_scalar(f"loss: {epoch}", loss.item(), global_step=epoch * l + i)
            logger.add_scalar(f"cl_error: {epoch}", cl_error, global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}")

        if epoch % 100 == 0:
            cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device)
            sampled_images = diffusion.sample(model, n=5, conditioning=cl)
            # pass sampled images through vae
            sampled_images = vae.decode(sampled_images)
            #plot_images(sampled_images)
            save_images_conditional(sampled_images, airfoil_x, os.path.join("results", args.run_name, f"{epoch}.jpg"), cl)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            wandb.log({"Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}.jpg"))],
                       "EMA Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))],
                       "learning_rate": current_lr})
    # save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join("results", args.run_name, "training_loss.jpg"))
    wandb.save(os.path.join("results", args.run_name, "training_loss.jpg"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="conditional_latent_run_4")
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--vae_model_path', type=str, default='vae_epoch_200.pt')
    parser.add_argument('--cond_dim', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    launch()



