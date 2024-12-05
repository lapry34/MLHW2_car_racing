import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd

from train_VAE import Encoder, Decoder, VAE, reparameterize

# Parameters
image_size = 96
latent_dim = 8
batch_size = 64

test_path = "../dataset/test"
train_path = "../dataset/train"

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

def compute_classwise_statistics(vae_model, dataset, latent_dim, save_path):
    """
    Computes mean and variance of latent variables for each class in the dataset.

    Args:
        vae_model: The trained VAE model.
        dataset: Dataset to analyze.
        latent_dim: Dimensionality of the latent space.
        save_path: Path to save the statistics as a CSV file.
    """
    encoder = vae_model.get_layer('encoder')  # Extract encoder from the VAE model
    z_means = []
    z_log_vars = []
    labels = []
    for batch_images, batch_labels in dataset:
        z_mean, z_log_var = encoder(batch_images)  # Retrieve z_mean and z_log_var
        z_means.extend(z_mean.numpy())  # Append z_mean values
        z_log_vars.extend(z_log_var.numpy())  # Append z_log_var values
        labels.extend(batch_labels.numpy())  # Collect the corresponding labels

    z_vars = np.exp(z_log_vars)  # Compute z_var from z_log_var
        
    # Compute class-wise statistics
    stats = []
    for c in np.unique(labels):
        indices = np.where(labels == c)[0]  # Find indices of samples belonging to class c

        # Extract z_mean and z_log_var for class c
        z_mean_c = np.array([z_means[i] for i in indices])
        z_var_c = np.array([z_vars[i] for i in indices])

        # Compute mean and variance for each dimension of the latent space
        mean = np.mean(z_mean_c, axis=0)
        variance = np.var(z_mean_c, axis=0) + np.mean(z_var_c, axis=0)   

        stats.append({"class": c, "mean": mean, "variance": variance})
    
    # Save statistics to CSV
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, "classwise_latent_stats.csv")
    
    # Convert to a DataFrame for better formatting and saving
    df = pd.DataFrame([
        {"class": stat["class"], 
         **{f"mean_{i}": stat["mean"][i] for i in range(latent_dim)},
         **{f"var_{i}": stat["variance"][i] for i in range(latent_dim)}}
        for stat in stats
    ])
    df.to_csv(csv_path, index=False)
    print(f"Class-wise latent statistics saved at: {csv_path}")

if __name__ == "__main__":
    # Load the trained VAE model
    vae = tf.keras.models.load_model("./vae_" + str(latent_dim) + ".keras", compile=False)

    # Load datasets
    train_dataset = load_dataset(path=train_path)
    test_dataset = load_dataset(path=test_path)

    # Print the model summary
    print(vae.summary())

    # Compute class-wise statistics for train dataset
    compute_classwise_statistics(vae, train_dataset, latent_dim, train_path + "_VAE_" + str(latent_dim))
    
    # Compute class-wise statistics for test dataset
    #compute_classwise_statistics(vae, test_dataset, latent_dim, test_path + "_VAE_" + str(latent_dim))