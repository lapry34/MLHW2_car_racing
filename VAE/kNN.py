import sys
sys.path.append("../")
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from metrics_plotter import generate_classification_report, generate_confusion_matrix
import pandas as pd

if __name__ == "__main__":
    num_classes = 5  # Number of classes in the dataset

    train_path = "../dataset/train_VAE/encoded_latents.csv"
    test_path = "../dataset/test_VAE/encoded_latents.csv"

    # Data Loading with pandas
    train = pd.read_csv(train_path).values
    test = pd.read_csv(test_path).values

    # last column is the label
    y_train = train[:, -1].astype(int)
    y_test = test[:, -1].astype(int)

    # remove the last column
    X_train = train[:, :-1]
    X_test = test[:, :-1]

    # Build kNN Classifier

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    print("kNN Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # Build another classifier using SVM
    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("SVM Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # do another classifier using Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))
    
    # do another classifier using Decision Tree
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # do another classifier using Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print("Gradient Boosting Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # do another classifier using AdaBoost
    from sklearn.ensemble import AdaBoostClassifier

    ab = AdaBoostClassifier()
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    print("AdaBoost Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # do another classifier using Extra Trees
    from sklearn.ensemble import ExtraTreesClassifier
    
    et = ExtraTreesClassifier()
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    print("Extra Trees Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    # do another classifier using Bagging of all the classifiers
    from sklearn.ensemble import VotingClassifier

    vc = VotingClassifier(estimators=[
        ("knn", knn),
        ("svm", svm),
        ("rf", rf),
        ("dt", dt),
        ("gb", gb),
        ("ab", ab),
        ("et", et)
    ])
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    print("Voting Classifier")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(num_classes)]))

    sys.exit(0)