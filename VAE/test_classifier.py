import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from train_VAE import Encoder

# Parameters
image_size = 96
latent_dim = 32
num_classes = 5  # Number of classes in the dataset
output_dir = "metrics_images"

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
    z_mean, _ = encoder(images)  # Use the encoder to extract latent features
    return z_mean.numpy()

# Main Execution
if __name__ == "__main__":
    # Load encoder and classifier
    if not os.path.exists("./encoder.keras"):
        raise FileNotFoundError("Encoder model not found. Please train and save the encoder first.")
    if not os.path.exists("./classifier.keras"):
        raise FileNotFoundError("Classifier model not found. Please train and save the classifier first.")
    encoder = tf.keras.models.load_model("./encoder.keras", compile=False)
    classifier = tf.keras.models.load_model("./classifier.keras", compile=False)

    # Load test dataset
    X_test, y_test = load_images_and_labels("../dataset/test")

    # Encode features using the encoder
    X_test_encoded = extract_features(encoder, X_test)

    # Prepare labels for evaluation
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes)

    # do prediction
    Y_pred = classifier.predict(X_test_encoded)

    # Evaluate the classifier by printing the classification report
    print("Classification Report:\n")
    print(classification_report(y_test_encoded, np.argmax(Y_pred, axis=1)))
    class_report = classification_report(y_test_encoded, np.argmax(Y_pred, axis=1), output_dict=True)
    class_names = list(class_report.keys())[:-3]

    # Convert classification report to tabular format
    header = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = [
        [class_name] + [
            f"{class_report[class_name][metric]:.2f}"
            for metric in ["precision", "recall", "f1-score"]
        ] + [f"{int(class_report[class_name]['support'])}"]
        for class_name in class_names
    ]

    # Add overall metrics
    rows.append(["Accuracy", "", "", f"{class_report['accuracy']:.2f}", ""])
    rows.append(["Macro Avg"] + [
        f"{class_report['macro avg'][metric]:.2f}" for metric in ["precision", "recall", "f1-score"]
    ] + [f"{int(class_report['macro avg']['support'])}"])
    rows.append(["Weighted Avg"] + [
        f"{class_report['weighted avg'][metric]:.2f}" for metric in ["precision", "recall", "f1-score"]
    ] + [f"{int(class_report['weighted avg']['support'])}"])

    # Save the Classification Report as an Image
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.5))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc='center',
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(header))))

    class_report_path = os.path.join(output_dir, "classification_report.png")
    plt.tight_layout()
    plt.savefig(class_report_path)
    print(f"Classification Report saved at: {class_report_path}")
    plt.close()

    conf_matrix = confusion_matrix(y_test_encoded, np.argmax(Y_pred, axis=1))

    # Plot and Save Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    print(f"Confusion Matrix saved at: {conf_matrix_path}")
    plt.close()
