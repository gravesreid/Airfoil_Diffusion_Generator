import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import AirfoilDataset
from conditionedDataset import ConditionedAirfoilDataset
from utils import *
from models import *
from datetime import datetime
from scipy.interpolate import CubicSpline
import neuralfoil as nf

# Create directory for saving plots
save_dir = os.path.expanduser(f"~/Downloads/gen_plots_{datetime.today().strftime('%Y-%m-%d')}/")

os.makedirs(save_dir, exist_ok=True)

airfoil_directory = 'generated_airfoils/'

airfoils = []

Dataset = AirfoilDataset(path = airfoil_directory)
ConditionedDataset = ConditionedAirfoilDataset(path = 'aero_sandbox_processed/', eval_path='clcd_data/')

uiuc_airfoils = []
for i in range(len(ConditionedDataset)):
    airfoil = ConditionedDataset.__getitem__(i)
    airfoil_dict = {"name": airfoil[0], "y": airfoil[1].cpu().numpy(), "ClCd": airfoil[2]}
    uiuc_airfoils.append(airfoil_dict)

uiuc_cl_values = [airfoil["ClCd"][0] for airfoil in uiuc_airfoils]
uiuc_cd_values = [airfoil["ClCd"][1] for airfoil in uiuc_airfoils]

for i in range(len(Dataset)):
    airfoil = Dataset.__getitem__(i)
    airfoil_dict = {"name": airfoil[0], "y": airfoil[1].cpu().numpy()}
    airfoils.append(airfoil_dict)

airfoil_x = Dataset.get_x().cpu().numpy()

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid') 

old_cl_list = []
old_cd_list = []
new_cl_list = []
new_cd_list = []

for i in range(len(airfoils)):
    original_airfoil = airfoils[i]
    original_airfoil_y = original_airfoil["y"].flatten()
    new_y = moving_average(original_airfoil_y, 3)

    new_coordinates = np.array([airfoil_x[2:], new_y]).T

    new_coef = nf.get_aero_from_coordinates(
        coordinates = new_coordinates,
        alpha = 0,
        Re = 1e6,
        model_size = "xxxlarge"
    )
    print(f"new cl = {new_coef['CL'][0]}, new cd = {new_coef['CD'][0]}")
    if new_coef['CL'][0] > -0.2:
        new_cl_list.append(new_coef['CL'][0])
    if new_coef['CD'][0] < 0.05:
        new_cd_list.append(new_coef['CD'][0])
    original_coordinates = np.array([airfoil_x, original_airfoil_y]).T
    original_coef = nf.get_aero_from_coordinates(
        coordinates = original_coordinates,
        alpha = 0,
        Re = 1e6,
        model_size = "xxxlarge"
    )
    print(f"original cl = {original_coef['CL'][0]}, original cd = {original_coef['CD'][0]}")
    if original_coef['CL'][0] > -0.2:
        old_cl_list.append(original_coef['CL'][0])
    if original_coef['CD'][0] < 0.05:
        old_cd_list.append(original_coef['CD'][0])

# plot histogram of new and old cl values
fig, (ax1) = plt.subplots( figsize=(10, 6))
ax1.hist(old_cl_list, bins=20, alpha=0.5, label='Original')
ax1.hist(new_cl_list, bins=20, alpha=0.5, label='Smoothed')
ax1.hist(uiuc_cl_values, bins=20, alpha=0.5, label='UIUC')
ax1.set_title(r"Generated $C_l$ values", fontsize=18)
ax1.set_xlabel(r"$C_l$", fontsize=16)
ax1.set_ylabel("Frequency", fontsize=16)
ax1.legend()
for spine in ax1.spines.values():
    spine.set_linewidth(2)

ax1.tick_params(axis='both', labelsize=16)

plt.tight_layout()
plt.show()

# plot histogram of new and old cd values
fig, (ax2) = plt.subplots(figsize=(10, 6))
ax2.hist(old_cd_list, bins=20, alpha=0.5, label='Original')
ax2.hist(new_cd_list, bins=20, alpha=0.5, label='Smoothed')
ax2.hist(uiuc_cd_values, bins=20, alpha=0.5, label='UIUC')
ax2.set_title(r"Generated $C_d$ values", fontsize=18)
ax2.set_xlabel(r"$C_d$", fontsize=16)
ax2.set_ylabel("Frequency", fontsize=16)
ax2.legend()
for spine in ax2.spines.values():
    spine.set_linewidth(2)
ax2.tick_params(axis='both', labelsize=16)

#plot the original airfoil with the new one
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(airfoil_x, original_airfoil_y, c='black', linewidth=2)
ax.plot(airfoil_x[2:], new_y, c='red', linewidth=2)
ax.set_title(f"Original airfoil vs. smoothed airfoil", fontsize=22)
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

