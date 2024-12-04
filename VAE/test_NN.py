import sys
sys.path.append("../")
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from metrics_plotter import generate_classification_report, generate_confusion_matrix
import matplotlib.pyplot as plt
from train_VAE import Encoder, reparameterize

# Parameters
image_size = 96
latent_dim = 32
num_classes = 5  # Number of classes in the dataset

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

# Main Execution
if __name__ == "__main__":
    # Load encoder and classifier
    if not os.path.exists("./encoder_" + str(latent_dim) + ".keras"):
        raise FileNotFoundError("Encoder model not found. Please train and save the encoder first.")
    if not os.path.exists("./classifier_" + str(latent_dim) + ".keras"):
        raise FileNotFoundError("Classifier model not found. Please train and save the classifier first.")
    encoder = tf.keras.models.load_model("./encoder_" + str(latent_dim) + ".keras", compile=False)
    classifier = tf.keras.models.load_model("./classifier_" + str(latent_dim) + ".keras", compile=False)

    # Load test dataset
    X_test, y_test = load_images_and_labels("../dataset/test")

    # Encode features using the encoder
    X_test_encoded = extract_features(encoder, X_test)

    # Prepare labels for evaluation
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

    # do prediction
    y_pred = classifier.predict(X_test_encoded)
    y_pred = np.argmax(y_pred, axis=1)

    # Get class names
    class_names = [str(i) for i in range(num_classes)]

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    generate_confusion_matrix(conf_matrix, class_names, "../metrics_plot/", f"conf_matrix_NN_{str(latent_dim)}.png")

    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report:\n")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    generate_classification_report(class_report, "../metrics_plot/", f"class_report_NN_{str(latent_dim)}.png")

    sys.exit(0)