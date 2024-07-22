import torch
import matplotlib.pyplot as plt
import os
from dataset import AirfoilDataset

def load_predictions(epoch, save_dir="predictions"):
    """
    Loads saved predictions for a given epoch.

    Parameters:
        epoch (int): The epoch number of the saved predictions.
        save_dir (str): Directory where predictions are saved.

    Returns:
        dict: A dictionary containing the loaded data, predictions, noise, and cl_cd.
    """
    save_path = os.path.join(save_dir, f"epoch_{epoch}_predictions.pt")
    checkpoint = torch.load(save_path)
    return checkpoint


def visualize_predictions(checkpoint, airfoil_x):
    """
    Visualizes the saved predictions.

    Parameters:
        checkpoint (dict): A dictionary containing the loaded data, predictions, noise, and cl_cd.
    """
    data = checkpoint['data']
    data = data.squeeze(1)
    predictions = checkpoint['predictions']
    predictions = predictions.squeeze(1)
    noise = checkpoint['noise']
    noise = noise.squeeze(1)
    cl_cd = checkpoint['cl_cd']
    noised_data =checkpoint['noised_data']
    noised_data = noised_data.squeeze(1)

    # Example visualization for a batch of predictions
    batch_size = data.size(0)
    num_examples = min(batch_size, 4)  # Number of examples to visualize

    fig, axes = plt.subplots(num_examples, 4, figsize=(15, num_examples * 5))
    for i in range(num_examples):
        axes[i, 0].scatter(airfoil_x, data[i].cpu().numpy(), label='Input', color='black')
        axes[i, 0].set_title(f"Input Data (Example {i + 1})")
        axes[i, 1].scatter(airfoil_x, predictions[i].cpu().numpy(), label='Prediction', color='black')
        axes[i, 1].set_title(f"Prediction (Example {i + 1})")
        axes[i, 2].scatter(airfoil_x, noise[i].cpu().numpy(), label='Noise', color='black')
        axes[i, 2].set_title(f"Noise (Example {i + 1})")
        axes[i, 3].scatter(airfoil_x, noised_data[i].cpu().numpy(), label='Noised Data', color='black')
        axes[i, 3].set_title(f"Noised Data (Example {i + 1})")
    
    plt.tight_layout()
    plt.show()



# test the function
epoch = 499
predictions = load_predictions(epoch)
print(f"Predictions keys: {predictions.keys()}")
print(f"Data shape: {predictions['data'].shape}")
print(f"predictions shape: { predictions['predictions'].shape }")
print(f"Noise shape: { predictions['noise'].shape }")
print(f"cl_cd shape: { predictions['cl_cd'].shape }")


# use dataloader to get x coordinates
dataset = AirfoilDataset(path="aero_sandbox_processed")
airfoil_x = dataset.get_x()

# visualize the predictions
visualize_predictions(predictions, airfoil_x)