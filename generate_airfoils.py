import pickle
from MakeDatasets import generate_airfoils_cd, generate_airfoils_cl

# Define the parameters for generating airfoils
num_airfoils = 2000  # Number of airfoils to generate
cd_range = (0.005, 0.02)  # Range of CD values
cl_range = (0.5, 1.5)  # Range of CL values

# Generate airfoils based on CD values
airfoils_cd = generate_airfoils_cd(num_airfoils, cd_range)

# Generate airfoils based on CL values
airfoils_cl = generate_airfoils_cl(num_airfoils, cl_range)

# Save the generated airfoils to .pkl files
with open('generated_airfoils_cd.pkl', 'wb') as f:
    pickle.dump(airfoils_cd, f)

with open('generated_airfoils_cl.pkl', 'wb') as f:
    pickle.dump(airfoils_cl, f)

print("Airfoils generated and saved to 'generated_airfoils_cd.pkl' and 'generated_airfoils_cl.pkl'")