import numpy as np
import matplotlib.pyplot as plt
import torch
import concurrent.futures
import os

def process_bands(band_path):
    """
    Chooses the middle frame of the bands, rounds up the DN values and converts them to 'uint8' type.
    """
    load_band = np.load(band_path)
    load_band = np.round(load_band[:, :, 4], 0) # Rounds off DN values correctly, changing type does not do that.
    load_band = load_band.astype(np.uint8)
    return load_band

def create_inputs(folder_id, path, set_type):
    """
    Combines all bands into a single file. Uses ThreadPoolExecutor for faster processing.
    """

    npy_filepath = os.path.join(path, folder_id)
    bands = sorted(os.listdir(npy_filepath))[:9]

    combined_bands = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_bands, os.path.join(npy_filepath, band)) for band in bands]
        combined_bands = [future.result() for future in concurrent.futures.as_completed(futures)]

    combined_bands = np.stack(combined_bands, axis=0)
    return combined_bands

# Metrics and Visualisation

def dice_coefficient(predicted, target):

    predicted = (predicted > 0.5).float()

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2.0 * intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return dice

def jaccard_index(predicted, target):

    predicted = (predicted > 0.5).float()

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    jaccard = (intersection) / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return jaccard

def visualize_segmentation(ground_truth, predicted):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].imshow(ground_truth, cmap='gray')
    axes[0].set_title('Ground Truth Mask')

    axes[1].imshow(predicted, cmap='gray')
    axes[1].set_title('Predicted Mask')

    for ax in axes:
        ax.axis('off')

    plt.show()