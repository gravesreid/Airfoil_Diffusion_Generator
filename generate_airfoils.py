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

def generate_airfoils(vae, diffusion_model, n_samples, cl=1, cd=0.02, device='cuda'):
    cl_values = torch.ones(n_samples, 1).to(device) * cl
    cd_values = torch.ones(n_samples, 1).to(device) * cd

    cl_cd_values = torch.cat((cl_values, cd_values), dim=1)
    
    print("cl_cd_values shape: ", cl_cd_values.shape)
    print("cl_cd_values: ", cl_cd_values)
    
    # Sample from the diffusion model
    sample, all_samples = diffusion_model.sampling(n_samples, cl_cd_values, device=device)
    sample = sample.squeeze()
    all_samples.squeeze()

    print("all_samples shape: ", all_samples.shape)
    print("sample shape: ", sample.shape)
    
    # Decode the sampled latent vectors into airfoil shapes using the VAE
    generated_airfoils = vae.decode(sample).detach().cpu().numpy()
    
    print("generated_airfoils shape: ", generated_airfoils.shape)

    # save generated airfoils
    dataset = AirfoilDataset(path='aero_sandbox_processed/')
    x_coords = dataset.get_x().cpu().numpy()
    for i in range(n_samples):
        airfoil = generated_airfoils[i]
        np.savetxt(f'generated_airfoils/generated_airfoil_cl{cl}_cd{cd}_{i}.dat', np.column_stack((x_coords, airfoil)), fmt='%.6f', delimiter=' ')

    
    return generated_airfoils


def plot_generated_airfoils(airfoil_path, eval_path, n_samples):
    # Load the generated airfoils
    airfoils = [f for f in os.listdir(airfoil_path) if f.endswith('.dat')]
    evals = [f for f in os.listdir(eval_path) if f.endswith('.csv')]
    
    # Ensure we have enough subplots
    rows = (n_samples + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(4 * rows, 16))
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
        
        axs[i].plot(x_coords, y_coords, c='black')
        axs[i].set_title(f"Generated Airfoil {i} - Cl/Cd: {cl_cd}")
        axs[i].set_aspect('equal')
        axs[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()

        
def generate_range_airfoils(vae, diffusion_model, n_samples, cl_range, cd_range, device='cuda'):
    cl_values = torch.linspace(cl_range[0], cl_range[1], n_samples).unsqueeze(1).to(device)
    cd_values = torch.linspace(cd_range[0], cd_range[1], n_samples).unsqueeze(1).to(device)

    cl_cd_values = torch.cat((cl_values, cd_values), dim=1)
    
    for cl, cd in cl_cd_values:
        generate_airfoils(vae, diffusion_model, n_samples, cl, cd, device)

 
def plot_distribution(eval_path, airfoil_path, title="Generated Airfoil Cl/Cd Distribution"):
    # Load the generated airfoils
    evals = [f for f in os.listdir(eval_path) if f.endswith('.csv')]
    airfoils = [f for f in os.listdir(airfoil_path) if f.endswith('.dat')]
    cl_cd = []
    
    unrealistic = []
    for eval in evals:
        with open(os.path.join(eval_path, eval), 'r', encoding="utf8", errors='ignore') as f:
            raw_data = f.readlines()
            cl_cd = raw_data[0].strip()  # Strip any extra whitespace
            cl, cd = cl_cd.split(',')
            cl, cd = float(cl), float(cd)
           # if cd > 1.0 or cl< 0.0 or cl > 2.0:
           #     unrealistic.append(eval.split('.')[0])
           #     os.remove(os.path.join(eval_path, eval))
            cl_cd = f"{cl:.3f}, {cd:.3f}"
            print(f"Cl/Cd: {cl_cd}")
        plt.scatter(cd, cl, c='black')

    for airfoil in airfoils:
        if airfoil.split('.')[0] in unrealistic:
            os.remove(os.path.join(airfoil_path, airfoil))
            print(f"Removed {airfoil}")
    print(f"Removed {len(unrealistic)} unrealistic airfoils")
        
    
    plt.title(title)
    plt.xlabel("Cd")
    plt.ylabel("Cl")
    plt.show()

# Number of samples to generate
n_samples = 16

plot_distribution("vae_recon_eval/", "vae_reconstructed/")

# Generate airfoils
#generated_airfoils = generate_airfoils(vae, diffusion_model, n_samples,cl=1.8, cd=0.02, device=device)

generate_range_airfoils(vae, diffusion_model, n_samples, cl_range=[0, .5], cd_range=[0.01, 0.02], device=device)
neural_foil_eval('generated_airfoils/', 'generated_eval/')
#plot_distribution("generated_eval/")
plot_distribution("generated_eval/", "generated_airfoils/")
# Plot generated airfoils
#plot_generated_airfoils("generated_airfoils/", "generated_eval/", n_samples)

