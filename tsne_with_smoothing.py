import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import aerosandbox as asb
import os
import torch
import neuralfoil as nf
from scipy.interpolate import interp1d, UnivariateSpline
from datetime import datetime

uiuc_path = 'aero_sandbox_processed/'
generated_path = 'generated_airfoils/'

uiuc_files = [f for f in os.listdir(uiuc_path) if f.endswith('.dat')]
generated_files = [f for f in os.listdir(generated_path) if f.endswith('.dat')]

uiuc_y_coords = []
generated_y_coords = []

uiuc_coords = []
generated_coords = []

# Define the number of points to interpolate to
num_points = 200

def interpolate_and_smooth_coords(x, y, num_points, smooth_factor=0.1):
    # Interpolates coordinates to a fixed number of points and smooths the y-coordinates
    f_interp = interp1d(x, y, kind='linear')
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = f_interp(x_new)
    
    # Apply cubic spline smoothing
    spline = UnivariateSpline(x_new, y_new, s=smooth_factor)
    y_smooth = spline(x_new)
    
    return x_new, y_smooth

# Process UIUC airfoils
for filename in uiuc_files:
    try:
        af = asb.Airfoil(name=filename[:-4], file=uiuc_path + filename)
        coords = af.coordinates
        x, y = coords[:, 0], coords[:, 1]
        x_new, y_smooth = interpolate_and_smooth_coords(x, y, num_points)
        uiuc_y_coords.append(y_smooth)
        uiuc_coords.append(np.column_stack((x_new, y_smooth)))
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        continue

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
        x_new, y_smooth = interpolate_and_smooth_coords(x_coord, y_coord, num_points)
        generated_y_coords.append(y_smooth)
        generated_coords.append(np.column_stack((x_new, y_smooth)))
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        continue

# Convert lists to numpy arrays
uiuc_y_coords = np.array(uiuc_y_coords)
generated_y_coords = np.array(generated_y_coords)

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

# Function to plot airfoils
def plot_airfoil(paired_list, airfoils_per_plot=48):
    print(f"Plotting {len(paired_list)} airfoils...")
    dat_str = datetime.now().strftime("%Y-%m-%d")
    directory = os.path.expanduser(f'~/Downloads/{dat_str}_generated_airfoil_plots/')

    os.makedirs(directory, exist_ok=True)

    n = len(paired_list)
    total_plots = (n + airfoils_per_plot - 1) // airfoils_per_plot  # Calculate the number of plots needed

    for plot_idx in range(total_plots):
        fig, axes = plt.subplots(8, 6, figsize=(30, 60))
        axes = axes.flatten()

        start_idx = plot_idx * airfoils_per_plot
        end_idx = min(start_idx + airfoils_per_plot, n)

        for ax_idx, airfoil_idx in enumerate(range(start_idx, end_idx)):
            airfoil_coords = paired_list[airfoil_idx]
            ax = axes[ax_idx]
            print(airfoil_coords)
            ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], c='black')
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"Airfoil {airfoil_idx}")

        # Hide any unused subplots
        for ax in axes[end_idx - start_idx:]:
            ax.axis('off')

        plt.tight_layout()
        file_path = os.path.join(directory, f'generated_airfoil_plot_{plot_idx}.png')
        #plt.savefig(file_path)
        #plt.close(fig)
        plt.show()

# Prepare the data for plotting

# Plot the airfoils
plot_airfoil(generated_coords, airfoils_per_plot=48)

