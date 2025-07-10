import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings

warnings.filterwarnings('ignore')


class AgglomerativeHierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering implementation with multiple linkage methods.
    Pure numpy implementation without scipy dependencies.
    """

    def __init__(self, n_clusters=2, linkage='single', metric='euclidean'):
        """
        Initialize the clustering algorithm.

        Parameters:
        n_clusters (int): Number of clusters to form
        linkage (str): Linkage criterion ('single', 'complete', 'average', 'ward')
        metric (str): Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None
        self.linkage_matrix_ = None
        self.cluster_centers_ = None

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance between two points."""
        return np.sum(np.abs(x1 - x2))

    def _cosine_distance(self, x1, x2):
        """Calculate cosine distance between two points."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
        return 1 - (dot_product / (norm_x1 * norm_x2))

    def _calculate_distance_matrix(self, X):
        """Calculate pairwise distance matrix."""
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.metric == 'euclidean':
                    dist = self._euclidean_distance(X[i], X[j])
                elif self.metric == 'manhattan':
                    dist = self._manhattan_distance(X[i], X[j])
                elif self.metric == 'cosine':
                    dist = self._cosine_distance(X[i], X[j])
                else:
                    raise ValueError(f"Unsupported metric: {self.metric}")

                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def _single_linkage(self, cluster1, cluster2, dist_matrix):
        """Single linkage: minimum distance between clusters."""
        min_dist = float('inf')
        for i in cluster1:
            for j in cluster2:
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
        return min_dist

    def _complete_linkage(self, cluster1, cluster2, dist_matrix):
        """Complete linkage: maximum distance between clusters."""
        max_dist = 0
        for i in cluster1:
            for j in cluster2:
                if dist_matrix[i, j] > max_dist:
                    max_dist = dist_matrix[i, j]
        return max_dist

    def _average_linkage(self, cluster1, cluster2, dist_matrix):
        """Average linkage: average distance between clusters."""
        total_dist = 0
        count = 0
        for i in cluster1:
            for j in cluster2:
                total_dist += dist_matrix[i, j]
                count += 1
        return total_dist / count if count > 0 else 0

    def _ward_linkage(self, cluster1, cluster2, X):
        """Ward linkage: minimize within-cluster variance."""
        centroid1 = np.mean(X[cluster1], axis=0)
        centroid2 = np.mean(X[cluster2], axis=0)

        n1, n2 = len(cluster1), len(cluster2)
        return np.sqrt((n1 * n2) / (n1 + n2)) * self._euclidean_distance(centroid1, centroid2)

    def _calculate_linkage_distance(self, cluster1, cluster2, dist_matrix, X):
        """Calculate distance between two clusters based on linkage method."""
        if self.linkage == 'single':
            return self._single_linkage(cluster1, cluster2, dist_matrix)
        elif self.linkage == 'complete':
            return self._complete_linkage(cluster1, cluster2, dist_matrix)
        elif self.linkage == 'average':
            return self._average_linkage(cluster1, cluster2, dist_matrix)
        elif self.linkage == 'ward':
            return self._ward_linkage(cluster1, cluster2, X)
        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")

    def fit(self, X):
        """
        Fit the hierarchical clustering algorithm to the data.

        Parameters:
        X (array-like): Input data of shape (n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # Calculate initial distance matrix
        dist_matrix = self._calculate_distance_matrix(X)

        # Store linkage information for dendrogram
        linkage_matrix = []
        merge_history = []

        # Perform agglomerative clustering
        step = 0
        while len(clusters) > self.n_clusters:
            min_distance = float('inf')
            merge_i, merge_j = 0, 1

            # Find the closest pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._calculate_linkage_distance(
                        clusters[i], clusters[j], dist_matrix, X
                    )

                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j

            # Merge the closest clusters
            merged_cluster = clusters[merge_i] + clusters[merge_j]

            # Store merge information
            merge_history.append({
                'step': step,
                'cluster1': clusters[merge_i].copy(),
                'cluster2': clusters[merge_j].copy(),
                'distance': min_distance,
                'size': len(merged_cluster)
            })

            # Store linkage information
            linkage_matrix.append([
                merge_i, merge_j, min_distance, len(merged_cluster)
            ])

            # Update clusters list
            new_clusters = []
            for k, cluster in enumerate(clusters):
                if k != merge_i and k != merge_j:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)
            clusters = new_clusters

            step += 1

        # Assign cluster labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                self.labels_[point_id] = cluster_id

        # Calculate cluster centers
        self.cluster_centers_ = []
        for cluster in clusters:
            center = np.mean(X[cluster], axis=0)
            self.cluster_centers_.append(center)
        self.cluster_centers_ = np.array(self.cluster_centers_)

        self.linkage_matrix_ = np.array(linkage_matrix)
        self.merge_history_ = merge_history
        return self

    def fit_predict(self, X):
        """
        Fit the algorithm and return cluster labels.

        Parameters:
        X (array-like): Input data

        Returns:
        labels (array): Cluster labels for each point
        """
        self.fit(X)
        return self.labels_

    def plot_dendrogram(self, X, figsize=(12, 8)):
        """
        Plot a simple dendrogram of the hierarchical clustering.

        Parameters:
        X (array-like): Input data
        figsize (tuple): Figure size
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must fit the model before plotting dendrogram")

        plt.figure(figsize=figsize)

        # Simple dendrogram visualization
        merge_distances = self.linkage_matrix_[:, 2]
        merge_steps = range(len(merge_distances))

        plt.plot(merge_steps, merge_distances, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Merge Step')
        plt.ylabel('Distance')
        plt.title(f'Hierarchical Clustering Merge Distances\nLinkage: {self.linkage}, Metric: {self.metric}')
        plt.grid(True, alpha=0.3)

        # Add horizontal line at the cut-off distance
        if len(merge_distances) > 0:
            cut_distance = merge_distances[-(self.n_clusters - 1)] if len(merge_distances) >= self.n_clusters - 1 else \
            merge_distances[-1]
            plt.axhline(y=cut_distance, color='red', linestyle='--', alpha=0.7,
                        label=f'Cut for {self.n_clusters} clusters')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_clusters(self, X, figsize=(10, 8)):
        """
        Plot the clustered data points.

        Parameters:
        X (array-like): Input data
        figsize (tuple): Figure size
        """
        if self.labels_ is None:
            raise ValueError("Must fit the model before plotting clusters")

        X = np.array(X)
        plt.figure(figsize=figsize)

        # Define colors for clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        # Plot points colored by cluster
        for i in range(self.n_clusters):
            cluster_mask = self.labels_ == i
            cluster_points = X[cluster_mask]

            if len(cluster_points) > 0:
                color = colors[i % len(colors)]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                            c=color, label=f'Cluster {i}', alpha=0.7, s=50)

        # Plot cluster centers
        if self.cluster_centers_ is not None:
            plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1],
                        c='black', marker='x', s=200, linewidths=3, label='Centroids')

        plt.title(f'Hierarchical Clustering Results\nLinkage: {self.linkage}, Metric: {self.metric}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_merge_process(self, X, figsize=(15, 10)):
        """
        Visualize the step-by-step merging process.

        Parameters:
        X (array-like): Input data
        figsize (tuple): Figure size
        """
        if not hasattr(self, 'merge_history_'):
            raise ValueError("Must fit the model before plotting merge process")

        X = np.array(X)
        n_steps = min(6, len(self.merge_history_))  # Show first 6 steps

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for step in range(n_steps):
            ax = axes[step]

            # Get clusters at this step
            current_clusters = [[i] for i in range(len(X))]

            # Apply merges up to current step
            for merge_step in range(step + 1):
                if merge_step < len(self.merge_history_):
                    merge_info = self.merge_history_[merge_step]
                    # This is a simplified visualization
                    # In practice, you'd need to track cluster evolution more carefully

            # Plot all points
            for i, point in enumerate(X):
                ax.scatter(point[0], point[1], c='lightgray', s=30, alpha=0.6)
                ax.annotate(str(i), (point[0], point[1]), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)

            merge_info = self.merge_history_[step]
            ax.set_title(f'Step {step + 1}: Merge distance = {merge_info["distance"]:.3f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def compare_linkage_methods(X, n_clusters=3):
    """Compare different linkage methods on the same dataset."""
    linkage_methods = ['single', 'complete', 'average', 'ward']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, linkage in enumerate(linkage_methods):
        clusterer = AgglomerativeHierarchicalClustering(
            n_clusters=n_clusters, linkage=linkage, metric='euclidean'
        )

        labels = clusterer.fit_predict(X)

        # Define colors
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # Plot results
        for j in range(n_clusters):
            cluster_mask = labels == j
            cluster_points = X[cluster_mask]
            if len(cluster_points) > 0:
                axes[i].scatter(cluster_points[:, 0], cluster_points[:, 1],
                                c=colors[j], label=f'Cluster {j}', alpha=0.7, s=50)

        # Plot centroids
        axes[i].scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1],
                        c='black', marker='x', s=200, linewidths=3)

        axes[i].set_title(f'{linkage.capitalize()} Linkage')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_sample_data():
    """Generate sample datasets for testing."""
    # Dataset 1: Well-separated blobs
    X1, y1 = make_blobs(n_samples=100, centers=3, cluster_std=1.0,
                        random_state=42, n_features=2)

    # Dataset 2: Circular data
    np.random.seed(42)
    theta = np.linspace(0, 2 * np.pi, 100)
    r1 = 2 + 0.5 * np.random.randn(50)
    r2 = 5 + 0.5 * np.random.randn(50)

    X2 = np.vstack([
        np.column_stack([r1[:50] * np.cos(theta[:50]), r1[:50] * np.sin(theta[:50])]),
        np.column_stack([r2[:50] * np.cos(theta[50:]), r2[:50] * np.sin(theta[50:])])
    ])

    return X1, y1, X2


# Example usage and demonstration
if __name__ == "__main__":
    print("=== Agglomerative Hierarchical Clustering Demo ===\n")

    # Generate sample data
    print("1. Generating sample data...")
    X1, y_true, X2 = generate_sample_data()

    # Basic usage
    print("\n2. Basic Hierarchical Clustering:")
    clusterer = AgglomerativeHierarchicalClustering(n_clusters=3, linkage='average')
    labels = clusterer.fit_predict(X1)
    print(f"Number of clusters formed: {len(np.unique(labels))}")
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(3)]}")

    # Plot results
    print("\n3. Visualizing clustering results...")
    clusterer.plot_clusters(X1)
    clusterer.plot_dendrogram(X1)

    # Compare different linkage methods
    print("\n4. Comparing different linkage methods:")
    compare_linkage_methods(X1, n_clusters=3)

    # Test different metrics
    print("\n5. Testing different distance metrics:")
    metrics = ['euclidean', 'manhattan', 'cosine']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric in enumerate(metrics):
        clusterer = AgglomerativeHierarchicalClustering(
            n_clusters=3, linkage='average', metric=metric
        )
        labels = clusterer.fit_predict(X1)

        colors = ['red', 'blue', 'green']
        for j in range(3):
            cluster_mask = labels == j
            cluster_points = X1[cluster_mask]
            if len(cluster_points) > 0:
                axes[i].scatter(cluster_points[:, 0], cluster_points[:, 1],
                                c=colors[j], label=f'Cluster {j}', alpha=0.7, s=50)

        axes[i].scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1],
                        c='black', marker='x', s=200, linewidths=3)
        axes[i].set_title(f'{metric.capitalize()} Distance')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Performance evaluation
    print("\n6. Cluster evaluation:")
    for linkage in ['single', 'complete', 'average', 'ward']:
        clusterer = AgglomerativeHierarchicalClustering(
            n_clusters=3, linkage=linkage
        )
        labels = clusterer.fit_predict(X1)

        try:
            ari = adjusted_rand_score(y_true, labels)
            silhouette = silhouette_score(X1, labels)
            print(f"{linkage.capitalize():>10} linkage - ARI: {ari:.3f}, Silhouette: {silhouette:.3f}")
        except Exception as e:
            print(f"{linkage.capitalize():>10} linkage - Error in evaluation: {e}")

    # Test on circular data
    print("\n7. Testing on circular data:")
    clusterer = AgglomerativeHierarchicalClustering(n_clusters=2, linkage='single')
    labels = clusterer.fit_predict(X2)
    clusterer.plot_clusters(X2)

    print("\n=== Hierarchical clustering demonstration completed! ===")