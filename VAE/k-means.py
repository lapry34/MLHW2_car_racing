import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("../")
from metrics_plotter import generate_classification_report, generate_confusion_matrix

latent_dim = 8

train_data_path = f"../dataset/train_VAE_{str(latent_dim)}/classwise_latent_stats.csv"
test_data_path = f"../dataset/test_VAE_{str(latent_dim)}/encoded_latents.csv"  # Replace with your actual test data path

def assign_cluster_labels(kmeans, train_samples, train_labels, n_clusters):
    """Assigns cluster labels based on the majority label in each cluster."""
    cluster_labels = kmeans.predict(train_samples)
    cluster_to_label = {}
    
    for cluster_id in range(n_clusters):
        cluster_points_labels = train_labels[cluster_labels == cluster_id]
        most_common_label = Counter(cluster_points_labels).most_common(1)[0][0]
        cluster_to_label[cluster_id] = most_common_label

    return cluster_to_label

def predict_with_kmeans(kmeans, cluster_to_label, test_samples):
    """Predicts the labels for test samples based on cluster assignments."""
    test_clusters = kmeans.predict(test_samples)
    predicted_labels = [cluster_to_label[cluster] for cluster in test_clusters]
    return np.array(predicted_labels)

if __name__ == "__main__":
    # Load the training CSV file
    data = pd.read_csv(train_data_path)
    
    # Generate training samples
    train_samples = []
    train_labels = []
    n_samples_per_class = 100

    for _, row in data.iterrows():
        class_label = int(row['class'])
        means = row[[f'mean_{i}' for i in range(latent_dim)]].values
        variances = row[[f'var_{i}' for i in range(latent_dim)]].values
        std_devs = np.sqrt(variances)
        generated_samples = np.random.normal(loc=means, scale=std_devs, size=(n_samples_per_class, latent_dim))
        train_samples.append(generated_samples)
        train_labels.extend([class_label] * n_samples_per_class)

    # Combine training samples into a single dataset
    train_samples = np.vstack(train_samples)
    train_labels = np.array(train_labels)

    # Perform K-means clustering
    n_clusters = len(data['class'].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=1979543)
    kmeans.fit(train_samples)

    # Assign cluster labels based on majority label
    cluster_to_label = assign_cluster_labels(kmeans, train_samples, train_labels, n_clusters)

    # Load the test data
    test_data = pd.read_csv(test_data_path)
    test_samples = test_data[[f'z_{i}' for i in range(latent_dim)]].values  # Replace with your test data format
    test_labels = test_data['label'].values

    # Predict test labels
    predicted_labels = predict_with_kmeans(kmeans, cluster_to_label, test_samples)
    
    print("\nClassification Report on Test Data:\n")
    print(classification_report(test_labels, predicted_labels))
    class_report = classification_report(test_labels, predicted_labels, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_k-means_{str(latent_dim)}.png")

    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    class_names = [str(i) for i in range(n_clusters)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot", f"conf_matrix_k-means_{str(latent_dim)}.png")

    #DO kNN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV

    # Apply standard scaling
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    param_grid = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9, 11]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted', verbose=3)
    grid.fit(train_samples, train_labels)
    best_knn = grid.best_estimator_
    print("Best Parameters from GridSearchCV:", grid.best_params_)
    k = grid.best_params_['kneighborsclassifier__n_neighbors']

    # Test on the test set
    y_pred = best_knn.predict(test_samples)
    print("\nClassification Report on Test Data:\n")
    print(classification_report(test_labels, y_pred))
    class_report = classification_report(test_labels, y_pred, output_dict=True)
    generate_classification_report(class_report, "metrics_plot/", f"class_report_generative_{k}NN_{str(latent_dim)}.png")

    conf_matrix = confusion_matrix(test_labels, y_pred)
    class_names = [str(i) for i in range(n_clusters)]
    generate_confusion_matrix(conf_matrix, class_names, "metrics_plot/", f"conf_matrix_generative_{k}NN_{str(latent_dim)}.png")

    sys.exit(0)