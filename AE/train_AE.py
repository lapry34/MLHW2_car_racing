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
            layers.Dense(latent_dim)
        ])

    def call(self, x):
        return self.encoder(x)

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

# AutoEncoder Model
@tf.keras.utils.register_keras_serializable()
class AutoEncoder(Model):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

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
def compute_loss(x, reconstructed):
    return tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstructed), axis=(1, 2)))

# Training Function
def train_autoencoder(autoencoder, dataset, optimizer, epochs):
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            reconstructed = autoencoder(x)
            loss = compute_loss(x, reconstructed)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
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


def a(tizio, *args):
    tizio(args)
# Main Execution
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset()

    # Check for existing models
    if os.path.exists("./autoencoder_" + str(latent_dim) + ".keras") and os.path.exists("./encoder_" + str(latent_dim) + ".keras") and os.path.exists("./decoder_" + str(latent_dim) + ".keras"):
        print("Loading existing models...")
        autoencoder = tf.keras.models.load_model("./autoencoder_" + str(latent_dim) + ".keras", compile=False)
        encoder = tf.keras.models.load_model("./encoder_" + str(latent_dim) + ".keras", compile=False)
        decoder = tf.keras.models.load_model("./decoder_" + str(latent_dim) + ".keras", compile=False)
    else:
        print("No existing models found. Initializing new models...")
        encoder = Encoder(latent_dim)
        decoder = Decoder(latent_dim)
        autoencoder = AutoEncoder(encoder, decoder)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Train the AutoEncoder
    epochs = 5
    train_autoencoder(autoencoder, dataset, optimizer, epochs)

    # Save models
    encoder.save("./encoder_" + str(latent_dim) + ".keras")
    decoder.save("./decoder_" + str(latent_dim) + ".keras")
    autoencoder.save("./autoencoder_" + str(latent_dim) + ".keras")

    sys.exit(0)