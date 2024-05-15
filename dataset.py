import os
import torch
import numpy as np
from torch.utils.data import Dataset

class AirfoilDataset(Dataset):
    def __init__(self, path='cleaned_airfoils/'):
        super(AirfoilDataset, self).__init__()
        self._X = torch.tensor([])  # Initialize as empty tensor
        self._Y = []  # List to hold tensors for each airfoil
        self.names = []  # Name of all airfoils
        self.norm_coeff = 0  # Normalization coeff to scale y to [-1, 1]

        airfoil_fn = [afn for afn in os.listdir(path) if afn.endswith('.dat')]
        all_y = []  # Temporary list to hold all y values for normalization

        # Process each airfoil file
        for idx, fn in enumerate(airfoil_fn):
            with open(os.path.join(path, fn), 'r', encoding="utf8", errors='ignore') as f:
                self.names.append(fn[:-4])  # Assuming .dat extension
                raw_data = f.readlines()
                x_coords, y_coords = [], []
                for line in raw_data:
                    parts = line.split()
                    if len(parts) == 2:
                        x, y = map(float, parts)
                        x_coords.append(x)
                        y_coords.append(y)
                if idx == 0:  # Only need to set _X once since x coordinates are shared
                    self._X = torch.tensor(x_coords, dtype=torch.float32)
                y_tensor = torch.tensor(y_coords, dtype=torch.float32)
                all_y.append(y_tensor)
                self._Y.append(y_tensor)

        # Determine normalization coefficient from all y values
        all_y_tensor = torch.cat(all_y)
        self.norm_coeff = torch.max(torch.abs(all_y_tensor))

        # Normalize each y tensor
        self._Y = [(y / self.norm_coeff) for y in self._Y]

    def __len__(self):
        return len(self._Y)

    def __getitem__(self, idx):
        # Return a clone of the tensor to prevent modifications to the original data
        return self.names[idx],self._Y[idx].clone().unsqueeze(0)

    def get_x(self):
        # Return a clone to ensure consistency with __getitem__
        return self._X.clone()

    def get_y(self):
        # Clone not strictly necessary here since it's a list of tensors, but shown for completeness
        return [y.clone() for y in self._Y]
