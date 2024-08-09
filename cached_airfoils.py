import pickle
import torch
import matplotlib.pyplot as plt

filepath = 'airfoil_cache.pkl'

with open(filepath, 'rb') as f:
    cache = pickle.load(f)
    coordinates = cache['coordinates']
    diffusion_training_coordinates = cache['diffusion_training_coordinates']
    CD = cache['CD']
    CL = cache['CL']


test_x_forward = torch.linspace(0, 1, int(len(diffusion_training_coordinates[0])/2))
test_x_backward = torch.linspace(1, 0, int(len(diffusion_training_coordinates[0])/2))
test_x = torch.hstack((test_x_backward, test_x_forward))
print('test_x shape:', test_x.shape)

def plot_test(original_points, test_points):
    original_y = original_points[:, 1]
    original_x = original_points[:, 0]

    fig, ax = plt.subplots()
    ax.plot(original_x, original_y, label='Original Airfoil')
    ax.plot(test_points, original_y, label='Test Airfoil')
    ax.legend()
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


plot_test(diffusion_training_coordinates[0], test_x)
