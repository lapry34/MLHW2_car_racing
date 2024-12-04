import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from train_VAE import Encoder, reparameterize

boosting = False
# Parameters
image_size = 96
latent_dim = 32
batch_size = 64
num_classes = 5  # Number of classes in the dataset

train_path = "../dataset/train"
test_path = "../dataset/test"

if boosting:
    train_path = "../dataset/train_boosted"

# Data Loading
def load_images_and_labels(base_path, folders=[0, 1, 2, 3, 4]):
    images = []
    labels = []
    for label in folders:
        folder_path = os.path.join(base_path, str(label))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=(image_size, image_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Feature Extraction with Encoder
def extract_features(encoder, images):
    z_mean_list, z_log_var_list = encoder(images)  # Use the encoder to extract latent features
    z_list = list()

    for z_mean, z_log_var in zip(z_mean_list, z_log_var_list):
        z = reparameterize(z_mean, z_log_var)
        z_list.append(z)
    
    z_list = np.array(z_list)

    #add a column of None before the first column
    return z_list
    #return z_mean_list.numpy()

# Build Feedforward Neural Network with Regularization
def build_classifier(input_dim, num_classes, _lambda=10e-4):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(_lambda)),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(_lambda))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'f1_score'])
    return model

# Main Execution
if __name__ == "__main__":
    # Load encoder
    if not os.path.exists("./encoder_" + str(latent_dim) + ".keras"):
        raise FileNotFoundError("Encoder model not found. Please train and save the encoder first.")
    encoder = tf.keras.models.load_model("./encoder_" + str(latent_dim) + ".keras", compile=False)

    # Load training dataset
    X_train, y_train = load_images_and_labels(train_path)

    # Encode features using the encoder
    X_train_encoded = extract_features(encoder, X_train)

    # Prepare labels for classification
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded, num_classes)

    # Load test dataset
    X_test, y_test = load_images_and_labels(test_path)
    X_test_encoded = extract_features(encoder, X_test)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded, num_classes)

    # Build and train the classifier
    classifier = build_classifier(latent_dim, num_classes)
    classifier.fit(X_train_encoded, y_train_categorical, epochs=250, batch_size=20, validation_data=(X_test_encoded, y_test_categorical))

    # Save the classifier
    classifier.save("./classifier_" + str(latent_dim) + ".keras")

    print("Classifier trained and saved successfully.")

    sys.exit(0)