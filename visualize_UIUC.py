import torch
import matplotlib.pyplot as plt
import numpy as np
from conditionedDataset import ConditionedAirfoilDataset
from utils import *
from models import *


airfoils = []

ConditionedDataset = ConditionedAirfoilDataset(path = 'aero_sandbox_processed/', eval_path='clcd_data/')

for i in range(1566):
    airfoil = ConditionedDataset.__getitem__(i)
    airfoil_dict = {"name": airfoil[0], "y": airfoil[1].cpu().numpy(), "ClCd": airfoil[2]}
    airfoils.append(airfoil_dict)

# order airfoils by Cl
airfoils = sorted(airfoils, key=lambda x: x["ClCd"][0])

# select 30 evenly spaced airfoils in descending order of Cl
airfoils = airfoils[::round(len(airfoils)/30)]
# reverse the order
airfoils = airfoils[::-1]

airfoil_x = ConditionedDataset.get_x().cpu().numpy()

fig, axes = plt.subplots(6, 5, figsize=(30, 50))

for i, ax in enumerate(axes.flat):
    ax.plot(airfoil_x, airfoils[i]["y"].flatten(), c='black')
    name = airfoils[i]["name"]
    clcd = airfoils[i]["ClCd"]
    cl = clcd[0]
    cd = clcd[1]
    cl = f"{cl:.3f}"
    cd = f"{cd:.3f}"
    ax.set_title(f"{name}, Cl: {cl}, Cd: {cd}")
    ax.axis('off')
    ax.set_aspect('equal')  
plt.show()