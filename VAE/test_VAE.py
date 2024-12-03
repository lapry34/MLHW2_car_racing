import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sys

from train_VAE import Encoder, Decoder, VAE

# Parameters
image_size = 96
latent_dim = 32
batch_size = 64


# Data Loading
def load_images_from_folders(base_path, folders):
    images = []
    for folder in folders:
        folder_path = os.path.join(base_path, str(folder))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=(image_size, image_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Load Dataset
def load_dataset():
    base_path = "../dataset/test"
    folders = [0, 1, 2, 3, 4]
    images = load_images_from_folders(base_path, folders)
    return tf.data.Dataset.from_tensor_slices(images).batch(batch_size)

# Test VAE
def test_vae(model, dataset, num_images=10):
    """
    Test the VAE by visualizing reconstructed images.

    Args:
        model: The trained VAE model.
        dataset: Dataset to test on.
        num_images: Number of images to visualize.
    """
    for batch in dataset.take(1):
        reconstructed, _, _ = model(batch)
        original_images = batch.numpy()
        reconstructed_images = reconstructed.numpy()

        # Visualize some original and reconstructed images
        fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
        for i in range(num_images):
            # Original images
            axes[0, i].imshow(original_images[i])
            axes[0, i].axis('off')
            # Reconstructed images
            axes[1, i].imshow(reconstructed_images[i])
            axes[1, i].axis('off')
        plt.suptitle("Top: Original Images, Bottom: Reconstructed Images")
        plt.show()

if __name__ == "__main__":
    # Load the trained VAE model
    vae = tf.keras.models.load_model("./vae.keras", compile=False)

    # Load the test dataset
    test_dataset = load_dataset()

    # Test the VAE
    test_vae(vae, test_dataset)
    sys.exit(0)