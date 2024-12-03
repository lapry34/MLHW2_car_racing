import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

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

# Compute Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot and Save Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
print(f"Confusion Matrix saved at: {conf_matrix_path}")
plt.close()

# Classification Report
class_report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

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