# Airfoil_Diffusion_Generator

Repo for an Airfoil generator that uses diffusion to create novel airfoil designs

Original data is from https://m-selig.ae.illinois.edu/ads/coord_database.html, stored in the zip file coord_selegFmt.zim

# TODO
- Write script to evaluate generated airfoils performance
- Optimize model to generate airfoils with desirable characteristics
- Literature review
- Write paper

# Completed
- Make U-net
- Make diffusion model
- Make github repository
- Get dataset and upload to repository
- Write script to clean data, interpolate to create 200 data points per airfoil. The Airfoils will share the x coordinates
- Verify airfoil plotting function