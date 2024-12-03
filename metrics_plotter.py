import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os 

def generate_classification_report(class_report, output_dir):

    class_names = list(class_report.keys())[:-3]

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

    # Save the classification report as a PNG image
    class_report_path = os.path.join(output_dir, "classification_report.png")
    plt.tight_layout()
    plt.savefig(class_report_path)
    print(f"Classification Report saved at: {class_report_path}")
    plt.close()

def generate_confusion_matrix(conf_matrix, class_names, output_dir):
    
    # Plot and Save Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    # Save the confusion matrix as a PNG image
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    print(f"Confusion Matrix saved at: {conf_matrix_path}")
    plt.close()