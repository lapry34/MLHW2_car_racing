import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    latent_dim = 4   # Replace with the desired latent dimension L
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

    # Define classifiers
    classifiers = {
        "kNN Classifier": KNeighborsClassifier(n_neighbors=1),
        "SVM Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Extra Trees Classifier": ExtraTreesClassifier(),
    }

    # Fit and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"{name}")
        print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ("knn", classifiers["kNN Classifier"]),
        ("svm", classifiers["SVM Classifier"]),
        ("rf", classifiers["Random Forest Classifier"]),
        ("dt", classifiers["Decision Tree Classifier"]),
        ("gb", classifiers["Gradient Boosting Classifier"]),
        ("ab", classifiers["AdaBoost Classifier"]),
        ("et", classifiers["Extra Trees Classifier"]),
    ])
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    print("Voting Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    sys.exit(0)