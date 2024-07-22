import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def performance_tsne(z_path, eval_path, perplexity=50, n_iter=5000):
    scaler = StandardScaler()
    z_list = []
    eval_list = []

    z_files = [f for f in os.listdir(z_path) if f.endswith('.csv')]
    eval_files = [f for f in os.listdir(eval_path) if f.endswith('.csv')]

    for i, (z_file, eval_file) in enumerate(zip(z_files, eval_files)):
        z_values = []
        with open(z_path + z_file, 'r', encoding="utf8") as f:
            raw_z = f.readlines()
            for line in raw_z:
                z = float(line.strip())
                z_values.append(z)
        z_list.append(z_values)
        
        with open(eval_path + eval_file, 'r', encoding="utf8") as f:
            raw_data = f.readlines()
            cl_cd = raw_data[0].strip()
            cl, cd = cl_cd.split(',')
            cl, cd = float(cl), float(cd)
            eval_list.append([cl, cd])


    z_vectors = torch.tensor(z_list).to(device)
    eval_array = np.array(eval_list)
    

    samples_cpu = z_vectors.cpu()
    samples_scaled = scaler.fit_transform(samples_cpu)

    pca = PCA(n_components=2)
    samples_pca_cpu = pca.fit_transform(samples_scaled)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    samples_tsne = tsne.fit_transform(samples_pca_cpu)
    
    def plot_samples(samples_tsne, color_values, title, color_label):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(samples_tsne[:, 0], samples_tsne[:, 1], c=color_values, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=color_label)
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot samples colored by cl values
    plot_samples(samples_tsne, eval_array[:, 0], 'Latent Space Visualization Colored by Coefficient of Lift (cl)', 'cl')

    # Plot samples colored by cd values
    plot_samples(samples_tsne, eval_array[:, 1], 'Latent Space Visualization Colored by Coefficient of Drag (cd)', 'cd')

z_path = 'vae_z/'
eval_path = 'vae_recon_eval/'

performance_tsne(z_path, eval_path, perplexity=100, n_iter=1000)
