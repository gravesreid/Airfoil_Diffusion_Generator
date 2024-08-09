import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from airfoil_dataset_1d import AirfoilDataset
from utils_1d import setup_logging
import logging

class Encoder(nn.Module):
    def __init__(self, airfoil_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(airfoil_dim, 256)
        self.fc2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc5_mu = nn.Linear(128, latent_dim)
        self.fc5_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        mu = self.fc5_mu(x)
        logvar = self.fc5_logvar(x)
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

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())   

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Update the model initialization to match the modified architecture
    model = VAE(input_channels=1, latent_dim=args.latent_dim, output_channels=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(dataloader):
            airfoil = data['train_coords'].to(device).float()  # Use 'train_coords' which is the reshaped 2D data
            print(f'Airfoil shape: {airfoil.shape}')
            optimizer.zero_grad()
            recon_airfoil, mu, logvar = model(airfoil)
            
            # Calculate the loss
            kl_loss = 0.1 * kl_divergence(mu, logvar)
            recon_loss = 10 * F.mse_loss(recon_airfoil, airfoil)
            loss = kl_loss + recon_loss
            
            loss.backward()
            optimizer.step()
            
            if i % args.log_interval == 0:
                logging.info(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, KL Loss: {kl_loss.item()}, Recon Loss: {recon_loss.item()}")
        
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f"vae_epoch_{epoch}.pt")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='vae')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_path', type=str, default='coord_seligFmt/')
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=100)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    launch()