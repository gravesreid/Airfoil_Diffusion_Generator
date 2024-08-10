import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
from airfoil_dataset_1d import AirfoilDataset
from torch.utils.data import DataLoader
import math

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-7, beta_end=1e-6, num_airfoil_points=200, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        #self.beta = self._cosine_variance_schedule(noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.num_airfoil_points = num_airfoil_points*2
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def _cosine_variance_schedule(self,timesteps=500,epsilon= 4e-5):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def noise_images(self, x, t, visualize=False):
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
            airfoil = airfoils[i].cpu().numpy()
            ax.scatter(airfoil_x, airfoil[0, :], color='black')
            ax.set_title(f'Timestep {i+1}')
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_steps = 500
    beta_start = 1e-7
    beta_end = 0.00005
    num_airfoil_points = 100  # Assuming 100 points per side, so 200 total

    # Initialize dataset and dataloader
    airfoil_path = 'coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize Diffusion model
    diffusion = Diffusion(noise_steps=noise_steps, beta_start=beta_start, beta_end=beta_end, num_airfoil_points=num_airfoil_points, device=device)
    
    # Get a single airfoil sample
    data = next(iter(dataloader))
    airfoil = data['train_coords_y'].to(device)
    airfoil_x = dataset.get_x()
    
    # Visualize the noising process
    n_timesteps_to_visualize = 50  # Number of timesteps to visualize
    timesteps_to_visualize = torch.linspace(0, noise_steps - 1, n_timesteps_to_visualize, dtype=torch.long).to(device)
    
    noised_airfoils = []
    
    for t in timesteps_to_visualize:
        t = t.view(1)  # Convert t to a batch of size 1
        noised_airfoil, _ = diffusion.noise_images(airfoil, t)
        noised_airfoils.append(noised_airfoil)

    # Convert list of noised airfoils to a tensor
    noised_airfoils = torch.cat(noised_airfoils, dim=0)
    
    # Plot the noised airfoils
    save_path = "noising_process_visualization.jpg"
    diffusion.plot_and_save_airfoils(noised_airfoils, airfoil_x, save_path)
    print(f"Noising process visualized and saved to {save_path}")

if __name__ == '__main__':
    main()
