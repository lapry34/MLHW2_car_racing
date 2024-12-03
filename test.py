import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import sys
from metrics_plotter import generate_classification_report, generate_confusion_matrix

if __name__ == "__main__":
    # Parameters
    input_size = (96, 96)
    batch_size = 32
    test_dir = "dataset/test"
    model_path = "cnn_model.keras"
    output_dir = "metrics_images"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Prepare the test data generator
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Get true labels and class indices
    true_labels = test_generator.classes
    class_indices = test_generator.class_indices
    class_names = list(class_indices.keys())

    # Predict on the test set
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    generate_confusion_matrix(conf_matrix, class_names, output_dir)

    # Classification Report
    class_report = classification_report(
        true_labels, predicted_labels, target_names=class_names, output_dict=True
    )
    print("Classification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))
    generate_classification_report(class_report, output_dir)
    sys.exit(0)