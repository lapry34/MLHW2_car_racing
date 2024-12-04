import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import sys
import pandas as pd

boosting = False
# Parameters
image_size = 96
latent_dim = 32
batch_size = 64
num_classes = 5  # Number of classes in the dataset

train_path = f"../dataset/train_AE_{latent_dim}/encoded_latents.csv"
test_path = f"../dataset/test_AE_{latent_dim}/encoded_latents.csv"

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

    # Data Loading with pandas
    train = pd.read_csv(train_path).values
    test = pd.read_csv(test_path).values

    # last column is the label
    y_train = train[:, -1].astype(int)
    y_test = test[:, -1].astype(int)

    # remove the last column
    X_train = train[:, :-1]
    X_test = test[:, :-1]

    # Apply standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare labels for classification
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded, num_classes)

    # Load test dataset
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded, num_classes)

    # Build and train the classifier
    classifier = build_classifier(latent_dim, num_classes)
    classifier.fit(X_train, y_train_categorical, epochs=250, batch_size=20, validation_data=(X_test, y_test_categorical))

    # Save the classifier
    classifier.save("./classifier_" + str(latent_dim) + ".keras")

    print("Classifier trained and saved successfully.")

    sys.exit(0)