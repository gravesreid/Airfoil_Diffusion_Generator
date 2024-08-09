import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import aerosandbox as asb
import matplotlib.pyplot as plt

class AirfoilDataset(Dataset):
    def __init__(self, airfoil_path, num_points_per_side=100, cache_file='airfoil_cache.pkl'):
        self.airfoil_path = airfoil_path
        self.airfoil_files = os.listdir(airfoil_path)
        self.airfoil_files = [f for f in self.airfoil_files if f.endswith(".dat")]
        self.num_points_per_side = num_points_per_side
        self.cache_file = cache_file
        
        if os.path.exists(self.cache_file):
            # Load cached data
            with open(self.cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.coordinates = cache['coordinates']
                self.diffusion_training_coordinates = cache['diffusion_training_coordinates']
                self.CD = cache['CD']
                self.CL = cache['CL']
        else:
            # Convert airfoils to airfoil objects
            self.airfoils = []
            for airfoil_file in self.airfoil_files:
                filepath = os.path.join(airfoil_path, airfoil_file)
                airfoil_name = airfoil_file.split(".")[0]
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    cleaned_lines = [line.strip() for line in lines if line.strip()]

                    # Remove header line if present
                    if not cleaned_lines[0][0].isdigit():
                        cleaned_lines = cleaned_lines[1:]

                    seen_lines = set()
                    unique_lines = []
                    # Remove duplicate points except for the first and last points
                    for i, line in enumerate(cleaned_lines):
                        if i == 0 or i == (len(cleaned_lines) - 1):
                            unique_lines.append(line)
                            seen_lines.add(line)
                        elif line not in seen_lines:
                            seen_lines.add(line)
                            unique_lines.append(line)
                    if len(cleaned_lines) != len(unique_lines):
                        print(f"Airfoil {airfoil_name} had duplicate points. Duplicate points were removed.")
                        with open(filepath, "w") as f:
                            print(f"Writing cleaned airfoil to {filepath}")
                            f.write("\n".join(unique_lines) + "\n")
                # Convert airfoil to airfoil object
                try:
                    airfoil = asb.Airfoil(
                        name=airfoil_name,
                        coordinates=filepath
                    )
                    self.airfoils.append(airfoil)
                except Exception as e:
                    print(f"Error loading airfoil {airfoil_name}: {e}")

            # Repanelize all airfoils
            self.repanelized_airfoils = []
            for airfoil in self.airfoils:
                try:
                    if len(airfoil.coordinates) < 2:
                        print(f"Skipping airfoil {airfoil.name} due to insufficient coordinates.")
                        continue
                    repanelized_airfoil = airfoil.repanel(n_points_per_side=num_points_per_side)
                    self.repanelized_airfoils.append(repanelized_airfoil)
                except Exception as e:
                    print(f"Error repaneling airfoil {airfoil.name}: {e}")

            # Assuming self.repanelized_airfoils is already defined and contains airfoils with coordinates
            self.coordinates = [airfoil.coordinates for airfoil in self.repanelized_airfoils]
            self.upper_coord = [airfoil.upper_coordinates() for airfoil in self.repanelized_airfoils]
            self.lower_coord = [airfoil.lower_coordinates() for airfoil in self.repanelized_airfoils]
            self.diffusion_training_coordinates = [np.vstack((upper, lower)) for upper, lower in zip(self.upper_coord, self.lower_coord)]

            self.CD = []
            self.CL = []
            for airfoil in self.repanelized_airfoils:
                print(f"Calculating CD and CL for airfoil {airfoil.name}")
                coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
                self.CL.append(coef['CL'][0])
                self.CD.append(coef['CD'][0])

            # Save to cache
            cache = {
                'coordinates': self.coordinates,
                'diffusion_training_coordinates': self.diffusion_training_coordinates,
                'CD': self.CD,
                'CL': self.CL
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)

    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        train_coords = self.diffusion_training_coordinates[idx]
        train_coords_y = train_coords[:, 1]  # only pass the y coordinates to the model
        train_coords_x = train_coords[:, 0]  # only pass the x coordinates to the model
        CD = self.CD[idx]
        CL = self.CL[idx]

        # Convert data to the required shape: (channels, data)
        train_coords = train_coords.T  # Transpose to shape (1, data_points)
        train_coords_y = train_coords_y.T  # Transpose to shape (1, data_points)

        coordinates = coordinates.T  # Transpose to shape (2, data_points)

        return {
            'train_coords': train_coords,
            'train_coords_y': train_coords_y,
            'CD': CD,
            'CL': CL
        }
    def get_x(self):
        return self.diffusion_training_coordinates[0][:, 0]

def plot_airfoil(airfoil):
    fig, ax = plt.subplots()
    ax.plot(airfoil[0,:], airfoil[1,:], color='black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

def plot_upper_and_lower_half(upper_half, lower_half):
    fig, ax = plt.subplots()
    ax.plot(upper_half[0,:], upper_half[1,:], color='black')
    ax.plot(lower_half[0,:], lower_half[1,:], color='black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

# Usage example
if __name__ == '__main__':
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # test dataloader
    for i, data in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Coordinates shape: {data['coordinates'].shape}")
        print(f"Train coordinates shape: {data['train_coords'].shape}")
        print(f"CD: {data['CD']}")
        print(f"CL: {data['CL']}")
        plot_airfoil(data['coordinates'][0])
        plot_airfoil(data['train_coords'][0])
        plot_upper_and_lower_half(data['upper_coord'][0], data['lower_coord'][0])
        break
    airfoil_sample = dataset[0]
    print("Coordinates shape:", airfoil_sample['coordinates'].shape)
    print("Upper coordinates shape:", airfoil_sample['upper_coord'].shape)
    print("Lower coordinates shape:", airfoil_sample['lower_coord'].shape)
    print("Train coordinates shape:", airfoil_sample['train_coords'].shape)
    print("CD:", airfoil_sample['CD'])
    print("CL:", airfoil_sample['CL'])
