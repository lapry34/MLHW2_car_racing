from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sys

if __name__ == "__main__":
    latent_dim = 8   # Replace with the desired latent dimension L
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
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf']
        }

        # Perform GridSearchCV with F1-macro as the scoring metric
        grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', verbose=3)
        grid.fit(X_train, y_train)

        # Use the best estimator from GridSearchCV
        best_svm = grid.best_estimator_
        print("Best Parameters from GridSearchCV:", grid.best_params_)
        print("Best Cross-Validation Score (F1-macro):", grid.best_score_)
    else:
        # Load specific parameters directly
        best_svm = SVC(C=10, gamma=0.1, kernel='rbf')  # Replace with desired parameters
        best_svm.fit(X_train, y_train)
        print("Using predefined SVM parameters: C=1, gamma=0.1, kernel='rbf'")

    # Test on the test set
    y_pred = best_svm.predict(X_test)
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))

    sys.exit(0)