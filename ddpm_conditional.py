import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, get_data, plot_images, save_images  # Ensure you have these functions in utils.py
from modules import UNet_conditional, EMA  # Ensure these are in modules.py
import logging
from airfoil_dataset import AirfoilDataset
from torch.utils.data import DataLoader
import wandb

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, num_airfoil_points=200, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.num_airfoil_points = num_airfoil_points
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, conditioning, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 2, self.num_airfoil_points)).to(self.device)
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
    
def chamfer_distance(pred, target):
    batch_size, num_points, _ = pred.shape
    pred = pred.unsqueeze(2).repeat(1, 1, num_points, 1)
    target = target.unsqueeze(1).repeat(1, num_points, 1, 1)
    dist = torch.norm(pred - target, dim=-1)
    min_dist_pred = torch.min(dist, dim=2)[0]
    min_dist_target = torch.min(dist, dim=1)[0]
    return torch.mean(min_dist_pred) + torch.mean(min_dist_target)



def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = UNet_conditional(c_in=2, c_out=2, cond_dim=args.cond_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(num_airfoil_points=args.num_airfoil_points, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # Wandb setup
    wandb.init(project='conditional_airfoil_diffusion', name=args.run_name, config=args)
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr


    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, airfoil in enumerate(pbar):
            train_coords = airfoil['train_coords'].to(device).float()
            cl = airfoil['CL'].to(device).float().unsqueeze(1)
            cd = airfoil['CD'].to(device)
            t = diffusion.sample_timesteps(train_coords.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(train_coords, t)
            if np.random.random() < 0.1:
                cl = None
            predicted_noise = model(x_t, t, cl)
            #loss = mse(noise, predicted_noise)
            loss = chamfer_distance(x_t, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(epoch_loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)

        if epoch % 5000 == 0:
            cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device)
            sampled_images = diffusion.sample(model, n=5, conditioning=cl)
            print(f'Sampled images shape: {sampled_images.shape}')
            ema_sampled_images = diffusion.sample(ema_model, n=5, conditioning=cl)  # Use cl instead of labels
            #plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"), cl)
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"), cl)
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            wandb.log({"Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}.jpg"))],
                       "EMA Generated Images": [wandb.Image(os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))]})



def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="DDPM_conditional")
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    launch()



