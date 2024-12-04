from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from metrics_plotter import generate_classification_report, generate_confusion_matrix
import os
import numpy as np

image_size = 96
num_classes = 5

train_path = "../dataset/train"

# Data Loading
def load_images_from_folders(base_path, folders=[0, 1, 2, 3, 4]):
    images = []
    labels = []
    for folder in folders:
        folder_path = os.path.join(base_path, str(folder))
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=(image_size, image_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(folder)  # Add folder label as the class
    return np.array(images), np.array(labels)
if __name__ == "__main__":

    # load the dataset
    images, labels = load_images_from_folders(train_path)

    # Reshape the images to 1D
    images = images.reshape(images.shape[0], -1)

    # Standardize the data
    scaler = StandardScaler()
    images_normalized = scaler.fit_transform(images)


    # Apply PCA
    pca = PCA(n_components=4)  # Set the number of principal components
    principal_components = pca.fit_transform(images_normalized)

    # Print the explained variance ratio
    sum_variances = pca.explained_variance_ratio_.sum()
    print("Explained Variances:", sum_variances)

    # do SVM
    from sklearn.svm import SVC

    # Load the test dataset
    test_path = "../dataset/test"
    X_test, y_test = load_images_from_folders(test_path)
    X_train, y_train = principal_components, labels

    # Reshape the images to 1D
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)

    # Train the SVM
    svm = SVC(kernel='rbf', C=1, gamma='auto')
    svm.fit(X_train, y_train)

    # Test the SVM
    y_pred = svm.predict(X_test)

    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_SVM_PCA-{str(pca.n_components_)}.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot/", f"conf_matrix_SVM_PCA-{str(pca.n_components_)}.png")

    # do KNN
    from sklearn.neighbors import KNeighborsClassifier

    # Train
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Test
    y_pred = knn.predict(X_test)

    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_KNN_PCA-{str(pca.n_components_)}.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot/", f"conf_matrix_KNN_PCA-{str(pca.n_components_)}.png")
    
    sys.exit(0)