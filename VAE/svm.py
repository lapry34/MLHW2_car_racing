from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sys
sys.path.append("../")
from metrics_plotter import generate_classification_report, generate_confusion_matrix

if __name__ == "__main__":
    latent_dim = 32   # Replace with the desired latent dimension L
    num_classes = 5  # Number of classes in the dataset

    train_path = f"../dataset/train_VAE_{latent_dim}/encoded_latents.csv"
    test_path = f"../dataset/test_VAE_{latent_dim}/encoded_latents.csv"

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

    # Flag for hyperparameter search
    perform_hyperparameter_search = False

    if perform_hyperparameter_search:
        # Define SVM classifier and parameter grid for hyperparameter search
        svm = SVC()
        param_grid = {
            'C': [0.1, 0.5, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 'auto', 'scale'],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }

        # Perform GridSearchCV with F1-macro as the scoring metric
        grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', verbose=3)
        grid.fit(X_train, y_train)

        # Use the best estimator from GridSearchCV
        best_svm = grid.best_estimator_
        print("Best Parameters from GridSearchCV:", grid.best_params_)
        print("Best Cross-Validation Score (F1-weighted):", grid.best_score_)
    else:
        # Load specific parameters directly
        best_svm = SVC(C=100, gamma='auto', kernel='rbf')  # Replace with desired parameters
        best_svm.fit(X_train, y_train)
        print("Using predefined SVM parameters: C=1, gamma=0.1, kernel='rbf'")

    # Test on the test set
    y_pred = best_svm.predict(X_test)
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_SVM_{str(latent_dim)}.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot/", f"conf_matrix_SVM_{str(latent_dim)}.png")

    sys.exit(0)