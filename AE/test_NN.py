import sys
sys.path.append("../")
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from metrics_plotter import generate_classification_report, generate_confusion_matrix

# Parameters
image_size = 96
latent_dim = 32
num_classes = 5  # Number of classes in the dataset

test_path = f"../dataset/test_AE_{latent_dim}/encoded_latents.csv"
train_path = f"../dataset/train_AE_{latent_dim}/encoded_latents.csv"


# Main Execution
if __name__ == "__main__":
    #load classifier
    if not os.path.exists("./classifier_" + str(latent_dim) + ".keras"):
        raise FileNotFoundError("Classifier model not found. Please train and save the classifier first.")
    classifier = tf.keras.models.load_model("./classifier_" + str(latent_dim) + ".keras", compile=False)

    # Load test dataset
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
    _ = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Prepare labels for classification
    label_encoder = LabelEncoder()
    _ = label_encoder.fit_transform(y_train)

    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded, num_classes)

    # do prediction
    y_pred = classifier.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Get class names
    class_names = [str(i) for i in range(num_classes)]

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot/", f"conf_matrix_NN_{str(latent_dim)}.png")

    # Classification Report
    class_report = classification_report(y_test_encoded, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report:\n")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    generate_classification_report(class_report, "metrics_plot/", f"class_report_NN_{str(latent_dim)}.png")

    sys.exit(0)