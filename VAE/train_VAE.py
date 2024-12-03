import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

def load_dataset(base_path="../dataset/train", folders=[0, 1, 2, 3, 4]):
    images = load_images_from_folders(base_path, folders)
    return tf.data.Dataset.from_tensor_slices(images).shuffle(1000).batch(batch_size)

# Define Encoder
@tf.keras.utils.register_keras_serializable()
class Encoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(image_size, image_size, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2 * latent_dim)
        ])

    def call(self, x):
        z_mean, z_log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def get_config(self):
        return {"latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Define Decoder
@tf.keras.utils.register_keras_serializable()
class Decoder(Model):
    def __init__(self, latent_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(12 * 12 * 64, activation='relu'),
            layers.Reshape((12, 12, 64)),
            layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, z):
        return self.decoder(z)

    def get_config(self):
        return {"latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Reparameterization Trick
def reparameterize(z_mean, z_log_var):
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return eps * tf.exp(z_log_var * 0.5) + z_mean

# VAE Model
@tf.keras.utils.register_keras_serializable()
class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def get_config(self):
        return {
            "encoder": tf.keras.utils.serialize_keras_object(self.encoder),
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }

    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.utils.deserialize_keras_object(config["encoder"])
        decoder = tf.keras.utils.deserialize_keras_object(config["decoder"])
        return cls(encoder=encoder, decoder=decoder)

# Loss Function
def compute_loss(x, reconstructed, z_mean, z_log_var):
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstructed), axis=(1, 2)))
    kl_divergence = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
    return reconstruction_loss + kl_divergence

# Training Function
def train_vae(vae, dataset, optimizer, epochs):
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = vae(x)
            loss = compute_loss(x, reconstructed, z_mean, z_log_var)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    for epoch in range(epochs):
        i = 0
        n_batches = len(list(dataset))
        for batch in dataset:
            if i % (n_batches // 4) == 0:
                print(f"Progress: {i // (n_batches // 4) * 25}%")
            loss = train_step(batch)
            i += 1
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Main Execution
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset()

    # Check for existing models
    if os.path.exists("./vae.keras") and os.path.exists("./encoder.keras") and os.path.exists("./decoder.keras"):
        print("Loading existing models...")
        vae = tf.keras.models.load_model("./vae.keras", compile=False)
        encoder = tf.keras.models.load_model("./encoder.keras", compile=False)
        decoder = tf.keras.models.load_model("./decoder.keras", compile=False)
    else:
        print("No existing models found. Initializing new models...")
        encoder = Encoder(latent_dim)
        decoder = Decoder(latent_dim)
        vae = VAE(encoder, decoder)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Train the VAE
    epochs = 500
    train_vae(vae, dataset, optimizer, epochs)

    # Save models
    encoder.save("./encoder.keras")
    decoder.save("./decoder.keras")
    vae.save("./vae.keras")

    sys.exit(0)