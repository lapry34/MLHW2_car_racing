from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sys
sys.path.append("../")
from metrics_plotter import generate_classification_report, generate_confusion_matrix

if __name__ == "__main__":
    latent_dim = 8   # Replace with the desired latent dimension L
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
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    param_grid = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9, 11]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted', verbose=3)
    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_
    print("Best Parameters from GridSearchCV:", grid.best_params_)
    k = grid.best_params_['kneighborsclassifier__n_neighbors']

    # Test on the test set
    y_pred = best_knn.predict(X_test)
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_pred))
    class_report = classification_report(y_test, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_kNN_{str(latent_dim)}.png")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(num_classes)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot", f"conf_matrix_kNN_{str(latent_dim)}.png")

    sys.exit(0)