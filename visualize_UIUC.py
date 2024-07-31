import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from conditionedDataset import ConditionedAirfoilDataset
from utils import *
from models import *
from datetime import datetime

# Create directory for saving plots
save_dir = os.path.expanduser(f"~/Downloads/UIUC_plots_{datetime.today().strftime('%Y-%m-%d')}/")
os.makedirs(save_dir, exist_ok=True)

airfoils = []

ConditionedDataset = ConditionedAirfoilDataset(path = 'aero_sandbox_processed/', eval_path='clcd_data/')

for i in range(1566):
    airfoil = ConditionedDataset.__getitem__(i)
    airfoil_dict = {"name": airfoil[0], "y": airfoil[1].cpu().numpy(), "ClCd": airfoil[2]}
    airfoils.append(airfoil_dict)

# sort airfoils by Cl
airfoils = sorted(airfoils, key=lambda x: x["ClCd"][0])
# reverse the order so that the highest Cl is first
airfoils = airfoils[::-1]

airfoil_x = ConditionedDataset.get_x().cpu().numpy()

# Save individual airfoil plots
for i, airfoil in enumerate(airfoils):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(airfoil_x, airfoil["y"].flatten(), c='black', linewidth=2)
    name = airfoil["name"]
    clcd = airfoil["ClCd"]
    cl = clcd[0]
    cd = clcd[1]
    cl = f"{cl:.2f}"
    cd = f"{cd:.3f}"
    ax.set_title(f"{name}, Cl: {cl}, Cd: {cd}", fontsize=22)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"uiuc_{i+1}.png"))
    plt.close(fig)

max_cl = max([airfoil["ClCd"][0] for airfoil in airfoils])
min_cl = min([airfoil["ClCd"][0] for airfoil in airfoils])

# Save histogram of Cl and Cd values
cl_values = [airfoil["ClCd"][0] for airfoil in airfoils]
cd_values = [airfoil["ClCd"][1] for airfoil in airfoils]
fig, (ax1) = plt.subplots( figsize=(10, 6))

ax1.hist(cl_values, bins=20, alpha=0.5)
ax1.set_title(r"UIUC $C_l$ values", fontsize=18)
ax1.set_xlabel(r"$C_l$", fontsize=16)
ax1.set_ylabel("Frequency", fontsize=16)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

ax1.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "histogram_cl.png"))
plt.close(fig)

fig, (ax2) = plt.subplots(figsize=(10, 6))
ax2.hist(cd_values, bins=20, alpha=0.5)
ax2.set_title(r"UIUC $C_d$ values", fontsize=18)
ax2.set_xlabel(r"$C_d$", fontsize=16)
ax2.set_ylabel("Frequency", fontsize=16)
for spine in ax2.spines.values():
    spine.set_linewidth(2)
ax2.tick_params(axis='both', labelsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "histogram_cd.png"))
plt.close(fig)

# Create a figure with 12 airfoils in 4 rows and 3 columns
fig, axes = plt.subplots(4, 3, figsize=(10, 8))
num_airfoils = len(airfoils)
step = num_airfoils // 12
for i in range(4):
    for j in range(3):
        idx = i * 3 + j
        airfoil = airfoils[idx * step]
        ax = axes[i, j]
        ax.plot(airfoil_x, airfoil["y"].flatten(), c='black', linewidth=2)
        name = airfoil["name"]
        clcd = airfoil["ClCd"]
        cl = clcd[0]
        cd = clcd[1]
        cl = f"{cl:.2f}"
        cd = f"{cd:.3f}"
        ax.set_title(f"{name} \n Cl: {cl} \n Cd: {cd}", fontsize=16)
        ax.axis('off')
        ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "uiuc_labeled.png"))
plt.close(fig)

def plot_all_airfoils(airfoil_dict, airfoil_x, airfoils_per_plot=32):
    num_airfoils = len(airfoil_dict)
    num_plots = int(np.ceil(num_airfoils / airfoils_per_plot))
    print(f"Plotting {num_airfoils} airfoils in {num_plots} plots")
    for i in range(num_plots):
        fig, axes = plt.subplots(4, 8, figsize=(10, 5))
        for j, ax in enumerate(axes.flat):
            idx = i * airfoils_per_plot + j
            if idx < num_airfoils:
                ax.plot(airfoil_x, airfoil_dict[idx]["y"].flatten(), c='black')
                name = airfoil_dict[idx]["name"]
                clcd = airfoil_dict[idx]["ClCd"]
                cl = clcd[0]
                cd = clcd[1]
                cl = f"{cl:.2f}"
                cd = f"{cd:.3f}"
                ax.set_title(f"{name} \n Cl: {cl} \n Cd: {cd}", fontsize=8)
                ax.axis('off')
                ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"uiuc_{i+1}.png"))
        plt.close(fig)

plot_all_airfoils(airfoils, airfoil_x)


# generated visualizations

