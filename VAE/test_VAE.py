import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

from train_VAE import Encoder, Decoder, VAE, reparameterize

# Parameters
image_size = 96
latent_dim = 8
batch_size = 64

test_path = "../dataset/test"
train_path = "../dataset/train"

save_encoded = True  # Change this flag to control saving encoded variables

# Data Loading
def load_images_from_folders(base_path, folders):
    images = []
    labels = []
    for folder in folders:
        folder_path = os.path.join(base_path, str(folder))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=(image_size, image_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(folder)  # Add folder label as the class
    return np.array(images), np.array(labels)

# Load Dataset
def load_dataset(path):
    folders = [0, 1, 2, 3, 4]
    images, labels = load_images_from_folders(path, folders)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)

def save_encoded_variables(vae_model, dataset, save_path):
    """
    Encodes the dataset using the VAE encoder and saves the latent variables (z_mean and z_log_var) into a CSV file.

    Args:
        vae_model: The trained VAE model.
        dataset: Dataset to encode.
        save_path: Path to save the encoded latent variables.
    """
    encoder = vae_model.get_layer('encoder')  # Extract encoder from the VAE model
    z_means = []
    z_log_vars = []
    labels = []
    z_list = []
    for batch_images, batch_labels in dataset:
        z_mean, z_log_var = encoder(batch_images)  # Retrieve z_mean and z_log_var
        z_means.extend(z_mean.numpy())  # Append z_mean values
        z_log_vars.extend(z_log_var.numpy())  # Append z_log_var values
        labels.extend(batch_labels.numpy())  # Collect the corresponding labels
        z = reparameterize(z_mean, z_log_var)  # Sample z from the distribution
        z_list.extend(z.numpy())

    # Create a DataFrame to save the encoded variables from z_list
    df = pd.DataFrame(z_list, columns=[f"z_{i}" for i in range(latent_dim)])
    df["label"] = labels

    # Save the DataFrame as a CSV
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, "encoded_latents.csv")
    df.to_csv(csv_path, index=False)
    print(f"Encoded variables saved at: {csv_path}")

    z_means = np.array(z_means)
    z_vars = np.array(np.exp(z_log_vars))  # Convert log variance to variance

    z_mean = np.mean(z_means, axis=0)
    z_var = np.mean(z_vars, axis=0) + np.var(z_means, axis=0)
    return z_mean, z_var

def generate_data(z_mean, z_var, vae, path, num_samples=1000):
    """
    Generate synthetic data using the mean and variance of the latent variables.

    Args:
        z_mean: Mean of the latent variables.
        z_var: Variance of the latent variables.
        vae: The trained VAE model.
    """
    z_samples = np.random.normal(z_mean, np.sqrt(z_var), size=(num_samples, latent_dim))
    z_samples = tf.convert_to_tensor(z_samples, dtype=tf.float32)

    # Load the decoder
    decoder = vae.decoder
    synthetic_images = decoder(z_samples)

    # Save synthetic images
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(synthetic_images):
        img_path = os.path.join(path, f"synthetic_{i}.png")
        plt.imsave(img_path, img.numpy())
    
    
    print(f"Synthetic images saved at: {path}")

# Test VAE with Iterative Visualization
def test_vae(model, dataset, num_images_per_class=1):
    """
    Test the VAE by visualizing reconstructed images. Shows an element of each class indefinitely.

    Args:
        model: The trained VAE model.
        dataset: Dataset to test on.
        num_images_per_class: Number of images to visualize per class.
    """
    all_images = []
    all_labels = []
    for batch_images, batch_labels in dataset:
        all_images.extend(batch_images.numpy())
        all_labels.extend(batch_labels.numpy())
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    unique_classes = np.unique(all_labels)

    while True:
        fig, axes = plt.subplots(2, len(unique_classes), figsize=(15, 5))
        for i, cls in enumerate(unique_classes):
            # Select a random image of the class
            class_indices = np.where(all_labels == cls)[0]
            selected_idx = np.random.choice(class_indices, size=num_images_per_class, replace=False)

            original_image = all_images[selected_idx[0]]
            reconstructed_image, _, _ = model(tf.expand_dims(original_image, axis=0))
            reconstructed_image = reconstructed_image.numpy()[0]

            # Original image
            axes[0, i].imshow(original_image)
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Original (Class {cls})")
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed_image)
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Reconstructed (Class {cls})")
        
        plt.suptitle("Top: Original Images, Bottom: Reconstructed Images")
        plt.show()

if __name__ == "__main__":
    # Load the trained VAE model
    vae = tf.keras.models.load_model("./vae_" + str(latent_dim) + ".keras", compile=False)

    # Load datasets
    train_dataset = load_dataset(path=train_path)
    test_dataset = load_dataset(path=test_path)

    # Check if we need to save the encoded variables
    if save_encoded:
        train_mean, train_var = save_encoded_variables(vae, train_dataset, train_path + "_VAE_" + str(latent_dim))
        test_mean, test_var = save_encoded_variables(vae, test_dataset, test_path + "_VAE_" + str(latent_dim))

        # Generate synthetic data
        num_samples = 1000
        generate_data(train_mean, train_var, vae, path=train_path + "_VAE_" + str(latent_dim) + "/synthetic")

    # Test the VAE
    try:
        test_vae(vae, test_dataset)
    except KeyboardInterrupt:
        print("Testing Interrupted.")
    sys.exit(0)