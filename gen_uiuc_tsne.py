import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import aerosandbox as asb
import os
import torch
import neuralfoil as nf
from scipy.interpolate import interp1d

uiuc_path = 'aero_sandbox_processed/'
generated_path = 'generated_airfoils/'

uiuc_files = [f for f in os.listdir(uiuc_path) if f.endswith('.dat')]
generated_files = [f for f in os.listdir(generated_path) if f.endswith('.dat')]

uiuc_coords = []
generated_coords = []

uiuc_ycoords = []
generated_ycoords = []


# Process UIUC airfoils
for filename in uiuc_files:
    x_coords, y_coords = [], []
    with open(os.path.join(uiuc_path, filename), 'r', encoding="utf8", errors='ignore') as f:
        raw_data = f.readlines()
        for line in raw_data:
            parts = line.split()
            if len(parts) == 2:
                x, y = map(float, parts)
                x_coords.append(x)
                y_coords.append(y)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        uiuc_ycoords.append(y_coords)
        uiuc_coords.append(np.stack((x_coords, y_coords), axis=1))


# Process generated airfoils
for i, filename in enumerate(generated_files):
    x_coord = []
    y_coord = []
    with open(os.path.join(generated_path, filename), 'r', encoding="utf8", errors='ignore') as f:
        raw_data = f.readlines()
        for line in raw_data:
            parts = line.split()
            if len(parts) == 2:
                x, y = map(float, parts)
                x_coord.append(x)
                y_coord.append(y)
    try:
        x_coord = np.array(x_coord)
        y_coord = np.array(y_coord)
        generated_ycoords.append(y_coord)
        generated_coords.append(np.stack((x_coord, y_coord), axis=1))
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        continue

# Convert lists to numpy arrays and flatten the coordinates for t-SNE
uiuc_coords = np.array([coords.flatten() for coords in uiuc_coords])
generated_coords = np.array([coords.flatten() for coords in generated_coords])

# Combine the coordinates
combined_coords = np.vstack((uiuc_coords, generated_coords))

# Create labels
labels = np.array(['UIUC'] * len(uiuc_coords) + ['Generated'] * len(generated_coords))

# Standardize the data
scaler = StandardScaler()
standardized_coords = scaler.fit_transform(combined_coords)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
tsne_components = tsne.fit_transform(standardized_coords)

# Plot the t-SNE components
plt.figure(figsize=(10, 7))
for label, color in zip(['UIUC', 'Generated'], ['red', 'blue']):
    plt.scatter(
        tsne_components[labels == label, 0],
        tsne_components[labels == label, 1],
        alpha=0.5,
        label=label,
        color=color
    )

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Airfoil Coordinates')
plt.legend()
plt.grid()
plt.show()

# Convert lists to numpy arrays
uiuc_y_coords = np.array(uiuc_ycoords)
generated_y_coords = np.array(generated_ycoords)

# Combine the y-coordinates
combined_y_coords = np.vstack((uiuc_y_coords, generated_y_coords))

# Create labels
labels = np.array(['UIUC'] * len(uiuc_y_coords) + ['Generated'] * len(generated_y_coords))

# Standardize the data
scaler = StandardScaler()
standardized_y_coords = scaler.fit_transform(combined_y_coords)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(standardized_y_coords)

# Plot the t-SNE components
plt.figure(figsize=(10, 7))
for label, color in zip(['UIUC', 'Generated'], ['red', 'blue']):
    plt.scatter(
        tsne_components[labels == label, 0],
        tsne_components[labels == label, 1],
        alpha=0.5,
        label=label,
        color=color
    )

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Airfoil Y-Coordinates')
plt.legend()
plt.grid()
plt.show()
