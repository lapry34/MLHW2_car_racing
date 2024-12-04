from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sys
sys.path.append("../")
from metrics_plotter import generate_classification_report, generate_confusion_matrix

if __name__ == "__main__":
    latent_dim = 32   # Replace with the desired latent dimension L
    num_classes = 5  # Number of classes in the dataset

    train_path = f"../dataset/train_AE_{latent_dim}/encoded_latents.csv"
    test_path = f"../dataset/test_AE_{latent_dim}/encoded_latents.csv"

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

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # Test on the test set
    y_pred = knn.predict(X_test)
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_kNN_{str(latent_dim)}.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot", f"conf_matrix_kNN_{str(latent_dim)}.png")

    sys.exit(0)