import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import AirfoilDataset
import math
from unet import Unet
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(airfoil_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3_mu = nn.Linear(128, latent_dim)
        self.fc3_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3_mu(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, airfoil_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class VAE(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(airfoil_dim, latent_dim)
        self.dec = Decoder(latent_dim, airfoil_dim)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)
    
class AirfoilDiffusion(nn.Module):
    def __init__(self,latent_size,in_channels,out_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8], device="cuda"):
        super().__init__()
        
        ###
        #  Part a: Your code here
        ###
        self.latent_size=latent_size
        self.in_channels=in_channels
        self.timesteps=timesteps

    

        self.beta=self._cosine_variance_schedule(timesteps).to(device)
        self.alpha=(1.0-self.beta).to(device)
        self.alpha_bar=torch.cumprod(self.alpha,dim=-1).to(device)
        self.alpha_prev_bar=torch.cat([torch.tensor([1.0],device=self.alpha.device),self.alpha_bar[:-1]],dim=0)
        self.sqrt_alpha_bar=torch.sqrt(self.alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar=torch.sqrt(1.0-self.alpha_bar).to(device)


        self.model=Unet(timesteps,time_embedding_dim,in_channels,out_channels,base_dim,dim_mults)

        
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
        

    def forward(self, x, noise, cl_cd):
        # Combine Cl and Cd into a single tensor
        cond = cl_cd
        target_time = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        #print("target time type in forward: ", target_time.type())
        #print("target time shape in forward: ", target_time.shape)
        #print("noise shape in forward: ", noise.shape)
        #print("cond shape in forward: ", cond.shape)
        x_t = self._forward_diffusion(x, target_time, noise, cond)
        #print("x_t shape in forward: ", x_t.shape)
        pred = self.model(x_t, target_time, cond)
        return pred

    
    
    def _forward_diffusion(self, x_0, t, noise, cond):
        Cl = cond[:, 0].unsqueeze(1).unsqueeze(2)  # Shape becomes [batch_size, 1, 1] compatible with x_0
        Cd = cond[:, 1].unsqueeze(1).unsqueeze(2)  # Same as above

        alpha_scaled = self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2)  # Ensure shape is [batch_size, 1, 1]
        one_minus_alpha_scaled = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2)  # Same as above

        q = alpha_scaled * x_0 * (1 + 0.1 * Cl) + one_minus_alpha_scaled * noise * (1 + 0.1 * Cd)

        return q


    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise, cl_cd):
        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        sqrt_one_minus_alpha_t_cumprod = self.sqrt_one_minus_alpha_bar[t]

        # Pass cl_cd as part of the model's prediction process
        #print("cl_cd shape: ", cl_cd.shape)
        #print("t type in reverse diffusion: ", t.type())
        #print("t shape in reverse diffusion: ", t.shape)
        #print("x_t shape in reverse diffusion: ", x_t.shape)
        prediction = self.model(x_t, t, cl_cd)  
        #print("prediction shape in reverse diffusion: ", prediction.shape)

        a = (1 / torch.sqrt(alpha_t)).reshape(x_t.shape[0], 1, 1)
        beta_t_reshaped = beta_t.view(-1, 1, 1)
        #print("beta_t_reshaped shape in reverse diffusion: ", beta_t_reshaped.shape)
        sqrt_one_minus_alpha_t_cumprod_reshaped = sqrt_one_minus_alpha_t_cumprod.view(-1, 1, 1)
        #print("sqrt_one_minus_alpha_t_cumprod_reshaped shape in reverse diffusion: ", sqrt_one_minus_alpha_t_cumprod_reshaped.shape)

        b = (x_t - (beta_t_reshaped / sqrt_one_minus_alpha_t_cumprod_reshaped) * prediction)
        #print("b shape in reverse diffusion: ", b.shape)
        mu = a * b
       #print("mu shape in reverse diffusion: ", mu.shape)

        if t.min() > 0:
            sigma = torch.sqrt(beta_t_reshaped)
        else:
            sigma = torch.tensor(0.0).to(x_t.device)

        return mu + sigma * noise
    @torch.no_grad()
    def sampling(self, n_samples, cl_cd_values, device="cuda"):
        # Initialize noise
        sample = torch.randn(n_samples, self.in_channels, self.latent_size, device=device)
        #print("sample shape in sampling: ", sample.shape)
        all_samples = [sample]

        # Assume cl_cd_values is a tensor of shape [n_samples, num_conditions]
        cl_cd_values = cl_cd_values.to(device)

        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.int64, device=device)
            #print("t_tensor shape in sampling: ", t_tensor.shape)
            sample = self._reverse_diffusion(sample, t_tensor, torch.randn_like(sample, device=device), cl_cd_values)
            sample.clamp_(-1, 1)
            all_samples.append(sample)

        all_samples = torch.stack(all_samples, dim=0)
        return sample.clone(), all_samples.clone()