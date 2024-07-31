import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from conditionedDataset import ConditionedAirfoilDataset
from dataset import AirfoilDataset
from utils import *
from models import *
import time
from datetime import datetime

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
diffusion_model = AirfoilDiffusion(airfoil_dim=airfoil_dim, in_channels=in_channels, out_channels=out_channels, device=device)
diffusion_model.load_state_dict(torch.load('diffusion.pth'))
diffusion_model.to(device)
diffusion_model.eval()


airfoil_dataset = AirfoilDataset(path='aero_sandbox_processed/')
airfoil_x = airfoil_dataset.get_x()


def visualize_generated_latent_space(samples_dict_list, perplexity=30, n_iter=1000, device='cuda'):
    scaler = StandardScaler()

    # Extract z vectors, cl, and cd from the dictionary list
    z_vectors = []
    for sample in samples_dict_list:
        z_vectors.append(sample['z'])
    cl_values = [sample['cl_cd'][0] for sample in samples_dict_list]
    cd_values = [sample['cl_cd'][1] for sample in samples_dict_list]

    # Concatenate all z vectors into a single tensor
    all_samples = torch.stack(z_vectors).to(device)
    print(f"all_samples shape: {all_samples.shape}")

    # Move samples to CPU and scale them
    samples_cpu = all_samples.cpu()
    samples_scaled = scaler.fit_transform(samples_cpu)
    samples_scaled_tensor = torch.tensor(samples_scaled, device=device)

    pca = PCA(n_components=2)

    # Apply PCA on CPU
    samples_pca_cpu = pca.fit_transform(samples_scaled)
    samples_pca = torch.tensor(samples_pca_cpu, device=device)

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    samples_tsne = tsne.fit_transform(samples_pca_cpu)

    # Function to plot the samples colored by a specific value
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
    plot_samples(samples_tsne, cl_values, 'Latent Space Visualization Colored by Coefficient of Lift (cl)', 'cl')

    # Plot samples colored by cd values
    plot_samples(samples_tsne, cd_values, 'Latent Space Visualization Colored by Coefficient of Drag (cd)', 'cd')




def generate_airfoils_latent(vae, diffusion_model, n_samples, cl=1, cd=0.02, device='cuda'):
    cl_values = torch.ones(n_samples, 1).to(device) * cl
    cd_values = torch.ones(n_samples, 1).to(device) * cd

    cl_cd_values = torch.cat((cl_values, cd_values), dim=1)
    
    print("cl_cd_values shape: ", cl_cd_values.shape)
    print("cl_cd_values: ", cl_cd_values)
    
    # Sample from the diffusion model
    sample, all_samples = diffusion_model.sampling(n_samples, cl_cd_values, device=device)
    sample = sample.squeeze()

    sample_numpy = sample.detach().cpu().numpy()
    print(f"sample_numpy shape: {sample_numpy.shape}")
    print(sample_numpy[:5])
    plt.hist(sample_numpy, bins=100)
    print(f"sample shape: {sample.shape}")

    # Decode the sampled latent vectors into airfoil shapes using the VAE
    generated_airfoils = vae.decode(sample).detach().cpu().numpy()
    
    print("generated_airfoils shape: ", generated_airfoils.shape)

    # save generated airfoils
    dataset = AirfoilDataset(path='aero_sandbox_processed/')
    x_coords = dataset.get_x().cpu().numpy()
    airfoil_list = []
    generated_dict_list = []
    good_airfoils = 0
    for i in range(n_samples):
        airfoil = generated_airfoils[i]
        coords = np.column_stack((x_coords, airfoil))
        airfoil_list.append(coords)
        z_vector = sample[i]
        np.savetxt(f'generated_airfoils/generated_airfoil_cl{cl}_cd{cd}_{i}.dat', coords, fmt='%.6f', delimiter=' ')
        cl_cd = get_coeff_from_coordinates(airfoil_dataset, airfoil)
        print(f"cl_cd: {cl_cd}")
        if cl_cd[0] > 0.0 and cl_cd[0] < 2.0 and cl_cd[1] < 0.05:
            airfoil_dict = {"coords":coords, "z":z_vector, "cl_cd":cl_cd}
            generated_dict_list.append(airfoil_dict)
            good_airfoils += 1
    print(f"Generated {good_airfoils} good airfoils")

    modulus = int(good_airfoils / 20)

    # create ordered list of airfoil coordinates based on cl values
    all_cl_cd = [sample['cl_cd'] for sample in generated_dict_list]
    all_coords = [sample['coords'] for sample in generated_dict_list]
    paired_list = list(zip(all_coords, all_cl_cd))
    paired_list.sort(key=lambda x: x[1][0])
    def plot_airfoil(paired_list):
        n = len(paired_list)
        for i, (coords, cl_cd) in enumerate(paired_list):
            if i % modulus == 0:
                print(f"coords shape: {coords.shape}")
                print(f"cl_cd: {cl_cd}")

                # Create a figure with three subplots
                fig, axes = plt.subplots(3, 4, figsize=(30, 50))

                # Define indices for the current and adjacent airfoils
                indices = [i, i + 1, i+2, i+3, i+4]
                
                for ax_idx, idx in enumerate(axes.flat):
                    airfoil_coords, airfoil_cl_cd = paired_list[idx]
                    cl = airfoil_cl_cd[0]
                    cd = airfoil_cl_cd[1]
                    ax = axes[ax_idx]
                    ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], c='black')
                    ax.set_aspect('equal')
                    ax.axis('off')
                    airfoil_cl_cd_str = f"{cl:.3f}, {cd:.3f}"
                    ax.set_title(f"Airfoil {idx} - Cl/Cd: {airfoil_cl_cd_str}")

                plt.savefig(f'generated_airfoils/generated_airfoil_{i}.png')
                plt.show()
    plot_airfoil(paired_list)
    #plot cl/cd distribution
    plt.figure(figsize=(10, 6))
    all_cl_cd = np.array(all_cl_cd)
    print("all_cl_cd shape: ", all_cl_cd.shape)
    plt.scatter(all_cl_cd[:,1], all_cl_cd[:,0], c='black')
    plt.title("Generated Airfoil Cl/Cd Distribution")
    plt.xlabel("Cd")
    plt.ylabel("Cl")
    plt.show()
    
    return generated_airfoils, sample, generated_dict_list, all_samples


def generate_airfoils(diffusion_model, n_samples, cl=1, cd=0.02, device='cuda'):
    cl_values = torch.ones(n_samples, 1).to(device) * cl
    cd_values = torch.ones(n_samples, 1).to(device) * cd

    cl_cd_values = torch.cat((cl_values, cd_values), dim=1)
    
    
    # Sample from the diffusion model
    sample, all_samples = diffusion_model.sampling(n_samples, cl_cd_values, device=device)
    sample = sample.squeeze()

    generated_airfoils = sample.detach().cpu().numpy()

    print("generated_airfoils shape: ", generated_airfoils.shape)

    # save generated airfoils
    dataset = AirfoilDataset(path='aero_sandbox_processed/')
    # load airfoil x coordinates
    x_coords = dataset.get_x().cpu().numpy()
    # load dataset airfoils
    UIUC_airfoils = []
    conditioned_dataset = ConditionedAirfoilDataset(path='aero_sandbox_processed/', eval_path='clcd_data/')
    for i in range(1566):
        airfoil = conditioned_dataset.__getitem__(i)
        UIUC_airfoils.append(airfoil[2].squeeze().numpy())
    airfoil_list = []
    generated_dict_list = []
    confidence_list = []
    cm_list = []
    xtr_top_list = []
    xtr_bot_list = []
    gen_outlier_list = []
    good_airfoils = 0
    for i in range(n_samples):
        airfoil = generated_airfoils[i]
        coords = np.column_stack((x_coords, airfoil))
        airfoil_list.append(coords)
        z_vector = sample[i]
        np.savetxt(f'generated_airfoils/generated_airfoil_cl{cl}_cd{cd}_{i}.dat', coords, fmt='%.6f', delimiter=' ')
        confidence, cl_cd, cm, top_xtr, bot_xtr = get_coeff_from_coordinates(airfoil_dataset, airfoil)
        np.savetxt(f'generated_eval/generated_airfoil_cl{cl}_cd{cd}_{i}.csv', cl_cd, delimiter=',')
        confidence_list.append(confidence)
        confidence = np.atleast_1d(confidence)
        np.savetxt(f'generated_confidence/generated_airfoil_confidence_cl{cl}_cd{cd}_{i}.csv', confidence)
        if confidence < 0.8:
            outlier_dict = {"name": f'generated_airfoil_cl{cl}_cd{cd}_{i}.dat', "confidence": confidence}
            gen_outlier_list.append(outlier_dict)
        # save the confidence, cl_cd, and xtr values from neuralfoil
        cm_list.append(cm)
        xtr_top_list.append(top_xtr)
        xtr_bot_list.append(bot_xtr)
        #print(f"cl_cd: {cl_cd}")
        if cl_cd[0] > -0.2 and cl_cd[0] < 1.9 and cl_cd[1] < .04:
            airfoil_dict = {"coords":coords, "z":z_vector, "cl_cd":cl_cd}
            generated_dict_list.append(airfoil_dict)
            good_airfoils += 1
    print(f"Generated {good_airfoils} good airfoils")

    # get the confidence values for uiuc airfoils
    uiuc_confidence_list = []
    uiuc_outlier_list = []
    path = 'aero_sandbox_processed/'
    files = [f for f in os.listdir(path) if f.endswith('.dat')]
    for filename in files:
        af = asb.Airfoil(name=filename[:-4], file=path + filename)
        coef = af.get_aero_from_neuralfoil(
                alpha=0,
                Re=2e6,
                mach=0.2,
            )
        confidence = coef['analysis_confidence'][0]
        if confidence < 0.8:
            outlier_dict = {"name": filename, "confidence": confidence}
            uiuc_outlier_list.append(outlier_dict)
        uiuc_confidence_list.append(confidence)
        cl = coef['CL'][0]
        cd = coef['CD'][0]

    # identify outliers in confidence values
    confidence_list = np.array(confidence_list)
    uiuc_confidence_list = np.array(uiuc_confidence_list)
    # get mean, std, and median of confidence values
    confidence_mean = np.mean(confidence_list)
    confidence_std = np.std(confidence_list)
    confidence_median = np.median(confidence_list)
    uiuc_confidence_mean = np.mean(uiuc_confidence_list)
    uiuc_confidence_std = np.std(uiuc_confidence_list)
    uiuc_confidence_median = np.median(uiuc_confidence_list)
    print("confidence mean: ", confidence_mean)
    print("confidence std: ", confidence_std)
    print("confidence median: ", confidence_median)
    print("uiuc confidence mean: ", uiuc_confidence_mean)
    print("uiuc confidence std: ", uiuc_confidence_std)
    print("uiuc confidence median: ", uiuc_confidence_median)
    # print outliers
    print("Generated Airfoil Outliers")
    for outlier in gen_outlier_list:
        print(outlier)
    print("UIUC Airfoil Outliers")
    for outlier in uiuc_outlier_list:
        print(outlier)



    # make box plot displaying confidence values

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot([confidence_list, uiuc_confidence_list], labels=['Generated', 'UIUC'])
    ax.set_xticklabels(['Generated', 'UIUC'])
    ax.set_ylabel("Confidence Value (0-1)")
    ax.set_title("Airfoil Evaluation Confidence Values")
    plt.show()

    # make a histogram of confidence values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence_list, bins=20, alpha=0.5, label='Generated')
    ax.hist(uiuc_confidence_list, bins=20, alpha=0.5, label='UIUC')
    ax.set_xlabel("Confidence Value (0-1)", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.set_title("Airfoil Evaluation Confidence Values", fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.legend()
    plt.tight_layout()
    plt.show()


    modulus = int(good_airfoils / 6)

    # create ordered list of airfoil coordinates based on cl values
    all_cl_cd = [sample['cl_cd'] for sample in generated_dict_list]
    all_coords = [sample['coords'] for sample in generated_dict_list]
    paired_list = list(zip(all_coords, all_cl_cd))
    paired_list.sort(key=lambda x: x[1][0])
    def plot_airfoil(paired_list, airfoils_per_plot=32):
        dat_str = datetime.now().strftime("%Y-%m-%d")
        directory = os.path.expanduser(f'~/Downloads/{dat_str}_generated_airfoil_plots/')

        os.makedirs(directory, exist_ok=True)

        n = len(paired_list)
        total_plots = (n + airfoils_per_plot - 1) // airfoils_per_plot  # Calculate the number of plots needed

        for plot_idx in range(total_plots):
            fig, axes = plt.subplots(4, 8, figsize=(10, 5))
            axes = axes.flatten()

            start_idx = plot_idx * airfoils_per_plot
            end_idx = min(start_idx + airfoils_per_plot, n)

            for ax_idx, airfoil_idx in enumerate(range(start_idx, end_idx)):
                airfoil_coords, airfoil_cl_cd = paired_list[airfoil_idx]
                cl = airfoil_cl_cd[0]
                cd = airfoil_cl_cd[1]
                ax = axes[ax_idx]
                ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], c='black')
                ax.set_aspect('equal')
                ax.axis('off')
                airfoil_cl= f"{cl:.2f}"
                airfoil_cd= f"{cd:.3f}"
                ax.set_title(f"Airfoil {airfoil_idx} \n  Cl: {airfoil_cl} \n Cd: {airfoil_cd}", fontsize=8)

            # Hide any unused subplots
            for ax in axes[end_idx - start_idx:]:
                ax.axis('off')

            plt.tight_layout()
            file_path = os.path.join(directory, f'gen_{plot_idx}.png')
            plt.savefig(file_path)
            plt.close(fig)

    plot_airfoil(paired_list)
    #plot cl/cd distribution
    plt.figure(figsize=(10, 6))

    # Convert lists to numpy arrays
    all_cl_cd = np.array(all_cl_cd)
    uiuc_cl_cd = np.array(UIUC_airfoils)

    print("all_cl_cd shape: ", all_cl_cd.shape)
    print("uiuc_cl_cd shape: ", uiuc_cl_cd.shape)

    # Find the overlap between generated and UIUC airfoils
    overlap_set = set(map(tuple, all_cl_cd)).intersection(set(map(tuple, uiuc_cl_cd)))
    overlap = np.array(list(overlap_set))

    # Ensure overlap is 2-dimensional
    if overlap.ndim == 1:
        overlap = np.expand_dims(overlap, axis=0)

    print("overlap shape: ", overlap.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(uiuc_cl_cd[:, 1], uiuc_cl_cd[:, 0], c='red', label='UIUC', alpha=0.5)
    ax.scatter(all_cl_cd[:, 1], all_cl_cd[:, 0], c='blue', label='Generated', alpha=0.5)
    if overlap.size > 0:
        ax.scatter(overlap[:, 1], overlap[:, 0], c='blue', label='Overlap')

    ax.set_title("Generated Airfoil Cl/Cd Distribution", fontsize=20)
    ax.set_xlabel(r"$C_d$", fontsize=18)
    ax.set_ylabel(r"$C_l$", fontsize=18)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(axis='both', labelsize=16)
    ax.legend()
    fig.tight_layout()

    plt.show()


    # make histogram of cl and cd values
    cl_values = [airfoil["cl_cd"][0] for airfoil in generated_dict_list]
    cd_values = [airfoil["cl_cd"][1] for airfoil in generated_dict_list]
    fig, (ax1) = plt.subplots( figsize=(10, 6))
    ax1.hist(cl_values, bins=20, alpha=0.5)
    ax1.set_title(r"Generated $C_l$ values", fontsize=20)
    ax1.set_xlabel(r"$C_l$", fontsize=18)
    ax1.set_ylabel("Frequency", fontsize=18)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.tick_params(axis = 'both', labelsize = 16)
    plt.tight_layout()
    plt.show()
    fig, (ax2) = plt.subplots(figsize=(10, 6))
    ax2.hist(cd_values, bins=20, alpha=0.5)
    ax2.set_title(r"Generated $C_d$ values", fontsize=20)
    ax2.set_xlabel(r"$C_d$", fontsize=18)
    ax2.set_ylabel("Frequency", fontsize=18)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.tick_params(axis = 'both', labelsize = 16)
    plt.tight_layout()
    plt.show()


    #make histogram with both generated and uiuc cl and cd values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(uiuc_cl_cd[:,0], bins=20, alpha=0.5, label=r'UIUC $C_l$')
    ax.hist(all_cl_cd[:,0], bins=20, alpha=0.5, label=r'Generated $C_l$')
    ax.set_title(r"$C_l$ Values of Generated and UIUC Airfoils", fontsize=20)
    ax.set_xlabel(r"$C_l$", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.legend()
    ax.tick_params(axis = 'both', labelsize = 16)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(uiuc_cl_cd[:,1], bins=20, alpha=0.5, label='UIUC Cd')
    ax.hist(all_cl_cd[:,1], bins=20, alpha=0.5, label='Generated Cd')
    ax.set_title(r"$C_d$ Values of Generated and UIUC Airfoils", fontsize=20)
    ax.set_xlabel(r"$C_d$", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.legend()
    plt.tight_layout()
    plt.show()

    save_dir = os.path.expanduser(f"~/Downloads/UIUC_sample{datetime.today().strftime('%Y-%m-%d')}/")
    os.makedirs(save_dir, exist_ok=True)

    # Sort generated airfoils by Cl
    generated_dict_list = sorted(generated_dict_list, key=lambda x: x["cl_cd"][0])
    # reverse the list so that the highest cl airfoils are first
    generated_dict_list = generated_dict_list[::-1]

    # Determine the spacing for even rows
    n_sections = 4
    section_size = len(generated_dict_list) // n_sections
    sections = [generated_dict_list[i*section_size:(i+1)*section_size] for i in range(n_sections)]

    # Create a plot with 12 generated airfoils ordered by Cl
    fig, axes = plt.subplots(4, 3, figsize=(10, 9))
    sample_number=0
    for i, section in enumerate(sections):
        for j, airfoil in enumerate(section):
            if j < 3:  # Only plot the first 3 airfoils in each section
                ax = axes[i, j]
                ax.plot(airfoil["coords"][:, 0], airfoil["coords"][:, 1], c='black', linewidth=2)
                cl = f"{airfoil['cl_cd'][0]:.2f}"
                cd = f"{airfoil['cl_cd'][1]:.3f}"
                ax.set_title(f"Airfoil Sample {sample_number + 1} \nCl: {cl} \n Cd: {cd}", fontsize=16)
                ax.axis('off')
                ax.set_aspect('equal')
                sample_number += 1

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gen_sample.png"))
    plt.close(fig)

    
    return generated_airfoils, sample, generated_dict_list, all_samples

def visualize_denoising(all_samples, airfoil_x):
    # Visualize the denoising process with all_samples
    num_samples = all_samples.size(0)
    num_examples = min(num_samples, 4)  # Number of examples to save


    for i in range(num_samples):
        plt.cla()
        plt.scatter(airfoil_x, all_samples[i][-1].cpu().numpy(), label='Input', color='black')
        plt.title(f"Denoised: (Example {i + 1})")
        plt.pause(.01)


# Number of samples to generate
n_samples = 1566

# clear the generated airfoils folder
for f in os.listdir('generated_airfoils/'):
    os.remove(os.path.join('generated_airfoils/', f))
# clear the generated eval folder
for f in os.listdir('generated_eval/'):
    os.remove(os.path.join('generated_eval/', f))

#plot_distribution("vae_recon_eval/", "vae_reconstructed/")

# Generate airfoils
generated_airfoil, sample, generated_dict_list, all_samples = generate_airfoils(diffusion_model, n_samples,cl=1.5, cd=0.003, device=device)
# visualize the denoising process with all_samples
#visualize_denoising(all_samples, airfoil_x)

print(f"samples shape: {sample.shape}")
# clear the generated airfoils folder

#_, sample_list = generate_range_airfoils(vae, diffusion_model, n_samples, cl_range=[1.9, 2.1], cd_range=[0.01, 0.02], device=device)
#visualize_generated_latent_space(generated_dict_list, perplexity=100, n_iter=500, device=device)
#neural_foil_eval('generated_airfoils/', 'generated_eval/')
#plot_distribution("generated_eval/")
#plot_distribution("generated_eval/", "generated_airfoils/")

# Plot generated airfoils
#plot_generated_airfoils("generated_airfoils/", "generated_eval/", n_samples)

