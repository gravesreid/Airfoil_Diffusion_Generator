import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import aerosandbox as asb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from airfoil_dataset import AirfoilDataset

# Define the diffusion model
class Simple2DDiffusionModel(nn.Module):
    def __init__(self):
        super(Simple2DDiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Usage example
if __name__ == '__main__':
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the model, criterion, and optimizer
    model = Simple2DDiffusionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):  # Number of epochs
        for i, data in enumerate(dataloader):
            train_coords = data['train_coords'].float()  # Assuming train_coords is in a tensor form
            noisy_sample = train_coords + torch.randn_like(train_coords) * 0.1  # Add noise
            noisy_sample = noisy_sample.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            output = model(noisy_sample)
            loss = criterion(output, train_coords.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the model weights
    torch.save(model.state_dict(), 'simple_2d_diffusion_model.pth')
