import aerosandbox as asb
import os
import numpy as np

path = 'coord_seligFmt/'
clcd_path = 'clcd_data/'
# List all .dat files in the specified directory
files = [f for f in os.listdir(path) if f.endswith('.dat')]
print(files[0])

# Iterate over each file
for filename in files:
    try:
        # Load the airfoil from a .dat file
        af = asb.Airfoil(name=filename[:-4], file=path + filename)
        print(af)
        
        # Repanel the airfoil
        af = af.repanel(n_points_per_side=100)
        af = af.normalize()
        coef = af.get_aero_from_neuralfoil(
                alpha=0,
                Re=2e6,
                mach=0.2,
            )
        cl = coef['CL'][0]
        cd = coef['CD'][0]
        combined_data = np.vstack((cl, cd)).T
        np.savetxt(clcd_path + filename[:-4] + '.csv', combined_data, delimiter=',')
        

        
        # Write the repaneled airfoil data back to a new .dat file
        af.write_dat(filepath='aero_sandbox_processed/' + filename)
        
        # Print the airfoil object and draw it
        print(af)
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        continue

