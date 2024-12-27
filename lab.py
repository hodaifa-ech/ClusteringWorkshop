import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



data = pd.read_csv("CC_GENERAL.csv")  # Ensure the file is in the same directory as lab.py

# Part 1: Data Visualization
# 1. Explore dataset
def explore_data(data):
    print("First five rows:")
    print(data.head())
    print("\nData info:")
    print(data.info())
    print("\nNull values:")
    print(data.isnull().sum())

explore_data(data)

print("\nStatistical summary:")
print(data.describe())

# 3. Scatter plot matrix
scatter_matrix(data.iloc[:, :10], alpha=0.2, figsize=(12, 12), diagonal='hist')
plt.suptitle("Scatter Matrix of Features")
plt.show()

# 4. PCA and TSNE
features = data.drop(columns=['CUST_ID'])  # Assuming 'CUST_ID' is the identifier column
features = features.fillna(features.mean())  # Handle missing values

# PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)
print(f"Explained Variance by PCA: {pca.explained_variance_ratio_}")
plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.6, cmap='viridis')
plt.title("PCA Transformation")
plt.show()
# TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], alpha=0.6, cmap='viridis')
plt.title("t-SNE Transformation")
plt.show()


# Part 2: Clustering
# 1. KMeans
# For PCA-based data
kmeans_pca = KMeans(n_clusters=3, random_state=42)
kmeans_pca_labels = kmeans_pca.fit_predict(pca_features)

# For TSNE-based data
kmeans_tsne = KMeans(n_clusters=3, random_state=42)
kmeans_tsne_labels = kmeans_tsne.fit_predict(tsne_features)
# 2. Elbow Method
# Define function to find optimal K
def elbow_method(data, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

elbow_method(pca_features, 10)
elbow_method(tsne_features, 10)
# 3. Cluster Visualization
def plot_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.show()

plot_clusters(pca_features, kmeans_pca_labels, "KMeans Clusters (PCA)")
plot_clusters(tsne_features, kmeans_tsne_labels, "KMeans Clusters (TSNE)")

# 4. Fuzzy C-means
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    pca_features.T, 3, 2, error=0.005, maxiter=1000, init=None
)
cluster_membership = np.argmax(u, axis=0)
plot_clusters(pca_features, cluster_membership, "Fuzzy C-means Clusters (PCA)")

# 5. DBSCAN
for eps in [0.5, 1.0]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(pca_features)
    plot_clusters(pca_features, dbscan_labels, f"DBSCAN Clusters (PCA) eps={eps}")

    # 6. Gaussian Mixture Model (EM)
em = GaussianMixture(n_components=3, random_state=42)
em_labels = em.fit_predict(pca_features)
plot_clusters(pca_features, em_labels, "EM Clusters (PCA)")

# 7. Hierarchical Clustering
linked = linkage(pca_features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

hc_labels = fcluster(linked, 3, criterion='maxclust')
plot_clusters(pca_features, hc_labels, "Hierarchical Clusters (PCA)")

# Conclusion
def compare_algorithms():
    algorithms = [
        ("KMeans PCA", silhouette_score(pca_features, kmeans_pca_labels)),
        ("KMeans TSNE", silhouette_score(tsne_features, kmeans_tsne_labels)),
        ("Fuzzy CMeans", silhouette_score(pca_features, cluster_membership)),
        ("EM", silhouette_score(pca_features, em_labels)),
    ]
    print("\nAlgorithm Comparison:")
    for name, score in algorithms:
        print(f"{name}: Silhouette Score = {score:.2f}")

compare_algorithms()
