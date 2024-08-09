import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from airfoil_dataset_2d import AirfoilDataset
from utils_2d import setup_logging
import logging

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64*8*8, 128)  # Adjust based on the size of the feature map
        self.fc2 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64*8*8)  # Adjust based on the size of the feature map
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 8, 8)  # Adjust based on the size of the feature map
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels):
        super(VAE, self).__init__()
        self.enc = Encoder(input_channels, latent_dim)
        self.dec = Decoder(latent_dim, output_channels)

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