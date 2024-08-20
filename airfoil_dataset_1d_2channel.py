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
                self.CM = cache['CM']
                self.max_camber = cache['max_camber']
                self.max_thickness = cache['max_thickness']
                self.TE_thickness = cache['TE_thickness']
                self.TE_angle = cache['TE_angle']
                self.names = cache['names']
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
            self.CM = []
            self.max_camber = []
            self.max_thickness = []
            self.TE_thickness = []
            self.TE_angle = []
            self.names = []
            for airfoil in self.repanelized_airfoils:
                print(f"Calculating CD and CL for airfoil {airfoil.name}")
                self.names.append(airfoil.name)
                coef = airfoil.get_aero_from_neuralfoil(alpha=0, Re=1e6, mach=0.0)
                print(f"CL: {coef['CL'][0]}, CD: {coef['CD'][0]}")
                max_camber = airfoil.max_camber()
                max_thickness = airfoil.max_thickness()
                TE_thickness = airfoil.TE_thickness()
                TE_angle = airfoil.TE_angle()
                self.CL.append(coef['CL'][0])
                self.CD.append(coef['CD'][0])
                self.CM.append(coef['CM'][0])
                self.max_camber.append(max_camber)
                self.max_thickness.append(max_thickness)
                self.TE_thickness.append(TE_thickness)
                self.TE_angle.append(TE_angle)

            # Save to cache
            cache = {
                'coordinates': self.coordinates,
                'diffusion_training_coordinates': self.diffusion_training_coordinates,
                'CD': self.CD,
                'CL': self.CL,
                'CM': self.CM,
                'max_camber': self.max_camber,
                'max_thickness': self.max_thickness,
                'TE_thickness': self.TE_thickness,
                'TE_angle': self.TE_angle,
                'names': self.names
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)

    def __len__(self):
        return len(self.coordinates)
    
    def __getitem__(self, idx):
        coordinates = self.coordinates[idx]
        train_coords = self.diffusion_training_coordinates[idx]
        train_coords_y = train_coords[:, 1]  # only pass the y coordinates to the model
        cd = self.CD[idx]
        cl = self.CL[idx]
        cm = self.CM[idx]
        max_camber = self.max_camber[idx]
        max_thickness = self.max_thickness[idx]
        TE_thickness = self.TE_thickness[idx]
        TE_angle = self.TE_angle[idx]
        name = self.names[idx]

        # Separate the y-coordinates into two parts
        train_coords_y_upper = train_coords_y[:self.num_points_per_side]  # First 100 points
        train_coords_y_lower = train_coords_y[self.num_points_per_side:]  # Second 100 points

        # Stack them to create a 2-channel tensor
        train_coords_y = np.stack([train_coords_y_upper, train_coords_y_lower], axis=0)

        # Convert data to the required shape: (channels, data)
        coordinates = coordinates.T  # Transpose to shape (2, data_points)

        return {
            'train_coords_y': torch.tensor(train_coords_y, dtype=torch.float32),
            'coordinates': torch.tensor(coordinates, dtype=torch.float32),
            'CD': cd,
            'CL': cl,
            'CM': cm,
            'max_camber': max_camber,
            'max_thickness': max_thickness,
            'TE_thickness': TE_thickness,
            'TE_angle': TE_angle,
            'name': name
        }

    def get_x(self):
        return self.diffusion_training_coordinates[0][:, 0]

def plot_airfoil(airfoil_x, airfoil):
    fig, ax = plt.subplots()
    y_coords = torch.cat([airfoil[0], airfoil[1]])
    ax.plot(airfoil_x, y_coords, color='black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


# Usage example
if __name__ == '__main__':
    airfoil_path = '/home/reid/Projects/Airfoil_Diffusion/denoising-diffusion-pytorch/coord_seligFmt'
    dataset = AirfoilDataset(airfoil_path, num_points_per_side=100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    airfoil_x = dataset.get_x()
    # test dataloader
    for i, data in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"Train coordinates shape: {data['train_coords_y'].shape}")
        print(f"CD: {data['CD']}")
        print(f"CL: {data['CL']}")
        print(f"Max Camber: {data['max_camber']}")
        print(f"Max Thickness: {data['max_thickness']}")
        plot_airfoil(airfoil_x,data['train_coords_y'][0])
        break
    airfoil_sample = dataset[0]
    print("Train coordinates shape:", airfoil_sample['train_coords_y'].shape)
    print("airfoil points:", airfoil_sample['train_coords_y'])
    print("CD:", airfoil_sample['CD'])
    print("CL:", airfoil_sample['CL'])
    print("Max Camber:", airfoil_sample['max_camber'])
    print("Max Thickness:", airfoil_sample['max_thickness'])
    print("TE Thickness:", airfoil_sample['TE_thickness'])
    print("TE Angle:", airfoil_sample['TE_angle'])


    # plotting histograms of CD, CL, Max Camber, and Max Thickness
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    axs = axs.flatten()
    axs[0].hist(dataset.CD, bins=20)
    axs[0].set_title("CD")
    axs[1].hist(dataset.CL, bins=20)
    axs[1].set_title("CL")
    axs[2].hist(dataset.max_camber, bins=20)
    axs[2].set_title("Max Camber")
    axs[3].hist(dataset.max_thickness, bins=20)
    axs[3].set_title("Max Thickness")
    axs[4].hist(dataset.TE_thickness, bins=20)
    axs[4].set_title("TE Thickness")
    axs[5].hist(dataset.TE_angle, bins=20)
    axs[5].set_title("TE Angle")
    axs[6].hist(dataset.CM, bins=20)
    axs[6].set_title("CM")
    plt.tight_layout()
    plt.show()
