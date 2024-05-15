import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import neuralfoil as nf

airfoil_name = "ah93w480b.dat"


coef1 = nf.get_aero_from_dat_file(
    "vae_reconstructed/" + airfoil_name,
    alpha=5,
    Re=1e6,
    model_size="xxxlarge",
)
coef2 = nf.get_aero_from_dat_file(
    "aero_sandbox_processed/" + airfoil_name,
    alpha=5,
    Re=1e6,
    model_size="xxxlarge",
)

print(f"Original: CL: {coef2['CL']}, CD: {coef2['CD']}")
print(f"Reconstructed: CL: {coef1['CL']} CD: {coef1['CD']}")

original_points = np.loadtxt("aero_sandbox_processed/" + airfoil_name, skiprows=1)
reconstructed_points = np.loadtxt("vae_reconstructed/" + airfoil_name)

original_x = original_points[:, 0]
original_y = original_points[:, 1]
reconstructed_x = reconstructed_points[:, 0]
reconstructed_y = reconstructed_points[:, 1]

plt.figure()
plt.plot(original_x, original_y, label="Original")
plt.plot(reconstructed_x, reconstructed_y, label="Reconstructed")
plt.gca().set_aspect("equal")
plt.legend()
plt.show()