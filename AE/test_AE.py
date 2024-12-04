import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

from train_AE import Encoder, Decoder, AutoEncoder  # Adjust imports to use AutoEncoder components

# Parameters
image_size = 96
latent_dim = 32
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

def save_encoded_variables(autoencoder, dataset, save_path):
    """
    Encodes the dataset using the AutoEncoder encoder and saves the latent variables into a CSV file.

    Args:
        autoencoder: The trained AutoEncoder model.
        dataset: Dataset to encode.
        save_path: Path to save the encoded latent variables.
    """
    encoder = autoencoder.encoder  # Extract encoder from the AutoEncoder model
    z_list = []
    labels = []
    for batch_images, batch_labels in dataset:
        z = encoder(batch_images)  # Retrieve latent vectors
        z_list.extend(z.numpy())
        labels.extend(batch_labels.numpy())  # Collect the corresponding labels

    # Create a DataFrame to save the latent variables
    df = pd.DataFrame(z_list, columns=[f"z_{i}" for i in range(latent_dim)])
    df["label"] = labels

    # Save the DataFrame as a CSV
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, "encoded_latents.csv")
    df.to_csv(csv_path, index=False)
    print(f"Encoded variables saved at: {csv_path}")

    return np.array(z_list)


# Test AutoEncoder with Iterative Visualization
def test_autoencoder(model, dataset, num_images_per_class=1):
    """
    Test the AutoEncoder by visualizing reconstructed images. Shows an element of each class indefinitely.

    Args:
        model: The trained AutoEncoder model.
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
            reconstructed_image = model(tf.expand_dims(original_image, axis=0))
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
    # Load the trained AutoEncoder model
    autoencoder = tf.keras.models.load_model("./autoencoder_" + str(latent_dim) + ".keras", compile=False)

    # Load datasets
    train_dataset = load_dataset(path=train_path)
    test_dataset = load_dataset(path=test_path)

    # Print the model summary
    print(autoencoder.summary())

    # Print the encoder and decoder summary
    print(autoencoder.encoder.summary())
    print(autoencoder.decoder.summary())

    # Check if we need to save the encoded variables
    if save_encoded:
        train_vectors = save_encoded_variables(autoencoder, train_dataset, train_path + "_AE_" + str(latent_dim))
        test_vectors = save_encoded_variables(autoencoder, test_dataset, test_path + "_AE_" + str(latent_dim))

    # Test the AutoEncoder
    try:
        test_autoencoder(autoencoder, test_dataset)
    except KeyboardInterrupt:
        print("Testing Interrupted.")
    sys.exit(0)