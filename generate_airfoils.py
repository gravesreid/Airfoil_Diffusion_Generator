import torch
import matplotlib.pyplot as plt
from dataset import AirfoilDataset
from utils import *
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define airfoil dimensions and latent size (ensure these match your training setup)
airfoil_dim = 199
latent_dim = 32
in_channels = 1
out_channels = 1

# Load the trained VAE model
vae = VAE(airfoil_dim, latent_dim)
vae.load_state_dict(torch.load('vae.pth'))
vae.to(device)
vae.eval()

# Load the trained diffusion model
diffusion_model = AirfoilDiffusion(latent_size=latent_dim, in_channels=in_channels, out_channels=out_channels, device=device)
diffusion_model.load_state_dict(torch.load('diffusion.pth'))
diffusion_model.to(device)
diffusion_model.eval()

def generate_airfoils(vae, diffusion_model, n_samples, device):
    # Generate a range of cl_cd values
    num_values = n_samples * 2
    linear_tensor = torch.linspace(.5, 1, num_values)
    cl_cd_values = linear_tensor.reshape(n_samples, 2).to(device)
    
    print("cl_cd_values shape: ", cl_cd_values.shape)
    print("cl_cd_values: ", cl_cd_values)
    
    # Sample from the diffusion model
    sample, all_samples = diffusion_model.sampling(n_samples, cl_cd_values, device=device)
    sample = sample.squeeze()
    
    print("sample shape: ", sample.shape)
    
    # Decode the sampled latent vectors into airfoil shapes using the VAE
    generated_airfoils = vae.decode(sample).detach().cpu().numpy()
    
    print("generated_airfoils shape: ", generated_airfoils.shape)

    # save generated airfoils
    dataset = AirfoilDataset(path='aero_sandbox_processed/')
    x_coords = dataset.get_x().cpu().numpy()
    for i in range(n_samples):
        airfoil = generated_airfoils[i]
        np.savetxt(f'generated_airfoils/generated_airfoil_{i}.dat', np.column_stack((x_coords, airfoil)), fmt='%.6f', delimiter=' ')

    
    return generated_airfoils

neural_foil_eval('generated_airfoils/', 'generated_eval/')

def plot_generated_airfoils(airfoil_path, eval_path, n_samples):
    # Load the generated airfoils
    airfoils = [f for f in os.listdir(airfoil_path) if f.endswith('.dat')]
    evals = [f for f in os.listdir(eval_path) if f.endswith('.csv')]
    
    # Ensure we have enough subplots
    rows = (n_samples + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(4 * rows, 4 * rows))
    axs = axs.flatten()  # Flatten in case of 2D array


    for i, (airfoil, eval) in enumerate(zip(airfoils, evals)):
        with open(os.path.join(airfoil_path, airfoil), 'r', encoding="utf8", errors='ignore') as f:
            raw_data = f.readlines()
            x_coords, y_coords = [], []
            for line in raw_data:
                parts = line.split()
                if len(parts) == 2:
                    x, y = map(float, parts)
                    x_coords.append(x)
                    y_coords.append(y)
        
        with open(os.path.join(eval_path, eval), 'r', encoding="utf8", errors='ignore') as f:
            raw_data = f.readlines()
            cl_cd = raw_data[0].strip()  # Strip any extra whitespace
            cl, cd = cl_cd.split(',')
            cl, cd = float(cl), float(cd)
            cl_cd = f"{cl:.3f}, {cd:.3f}"
            print(f"Cl/Cd: {cl_cd}")
        
        axs[i].plot(x_coords, y_coords)
        axs[i].set_title(f"Generated Airfoil {i} - Cl/Cd: {cl_cd}")
        axs[i].set_aspect('equal')
        axs[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()

        

 

# Number of samples to generate
n_samples = 8

# Generate airfoils
generated_airfoils = generate_airfoils(vae, diffusion_model, n_samples, device)

# Plot generated airfoils
plot_generated_airfoils("generated_airfoils/", "generated_eval/", n_samples)

