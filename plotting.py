import matplotlib.pyplot as plt
import numpy as np
import os

def plot_scalogram_with_fill(coefs, index, save_dir, label):
    """Plots and saves scalograms for one set of coefficients."""
    fig, axes = plt.subplots(16, 16, figsize=(8, 8))

    # Iterate over channels (coefs has shape (247, 100, 241))
    for ch_idx, channel in enumerate(coefs):                 
        r, c = divmod(ch_idx, 16)
        axes[r, c].pcolormesh(np.abs(channel), cmap='viridis')
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    # Fill remaining empty subplots with blank images
    for idx in range(coefs.shape[0], 16 * 16):
        r, c = divmod(idx, 16)
        axes[r, c].pcolormesh(np.zeros((50, 241)), cmap='viridis')
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    # Hide subplot borders
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    fig.patch.set_visible(False)

    # Save image
    save_path = os.path.join(save_dir, f"{label}_full_channel_{index}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Close figure to free memory


def plot_average_scalograms(coefs, index, save_dir, label):
    
    fig, axes = plt.subplots(figsize=(8,8))
    
    average = np.mean(np.abs(coefs), axis=0)
    axes.pcolormesh(average)

    axes.set_xticks([])
    axes.set_yticks([])
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    fig.patch.set_visible(False)

    # Save image
    save_path = os.path.join(save_dir, f"{label}_average_channel_{index}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Close figure to free memory



cwd = os.getcwd()
data_dir = cwd + "/saved_data/"
files = ['a_coefficients_18_2_Feb10.npy', 'e_coefficients_18_2_Feb10.npy', 'i_coefficients_18_2_Feb10.npy']

##Plotting Loops

for file in files:
    coefs = np.load(data_dir + file, mmap_mode='r')
    save_dir = cwd + f"/scalograms_test/images/"
    for i in range(coefs.shape[0]):
            print(f"Processing slice {i+1}/{coefs.shape[0]}")
            plot_scalogram_with_fill(coefs = coefs[i], index = i, save_dir = save_dir, label=file[0])  # Load and process only one slice
    print(f"{file[0]} scalograms completed")


for file in files:
    coefs = np.load(data_dir + file, mmap_mode='r')
    save_dir = cwd + f"/scalograms_test/images_average/"
    for i in range(coefs.shape[0]):
            print(f"Processing slice {i+1}/{coefs.shape[0]}")
            plot_average_scalograms(coefs = coefs[i], index = i, save_dir = save_dir, label=file[0])  # Load and process only one slice
    print(f"{file[0]} scalograms completed")