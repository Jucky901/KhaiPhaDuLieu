import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#   NHÓM 2
#Trương Quốc Vương - 22110460 - 50%
#Nguyễn Đức Tín - 22110434 - 50% 

class KMeans:
    def __init__(self, k=3, max_iters=500, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """ Randomly initialize centroids from the dataset """
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.permutation(len(X))
        self.centroids = X[random_indices[:self.k]]
    
    def compute_distance(self, X, centroids):
        """ Compute the Euclidean distance between data points and centroids """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances

    def assign_clusters(self, distances):
        """ Assign each data point to the nearest centroid """
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        """ Update centroids by calculating the mean of each cluster """
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        """ Fit the K-Means model to the data """
        self.initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = self.compute_distance(X, self.centroids)
            self.labels = self.assign_clusters(distances)

            new_centroids = self.update_centroids(X, self.labels)

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """ Predict the cluster for each data point """
        distances = self.compute_distance(X, self.centroids)
        return self.assign_clusters(distances)

    def plot_clusters(self, X):
        """ Plot the clusters and centroids """
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}')
        
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

# Load dataset
data = pd.read_csv('DataMining/kmeans_data_with_initial_nodes.csv')

# Extract features (x1, x2)
X = data[['x1', 'x2']].values

# Apply K-Means
k = 3  # Number of clusters
kmeans = KMeans(k=k)
kmeans.fit(X)

# Plot the resulting clusters


# Predict new data points
new_points = np.array([[5, 5], [10, 10], [3, 15], [7,1]])
predictions = kmeans.predict(new_points)
print("New Data Points Predictions:", predictions + 1)

kmeans.plot_clusters(X)