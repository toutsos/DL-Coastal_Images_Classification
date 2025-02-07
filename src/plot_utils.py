import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score




def plot_metrics(log_dir: str):
    metrics_csv = f"{log_dir}/metrics.csv"
    metrics_df = pd.read_csv(metrics_csv)

    if "loss" in metrics_df.columns and "val_loss" in metrics_df.columns:
        trn_loss_df = metrics_df[["epoch", "loss"]].dropna()
        val_loss_df = metrics_df[["epoch", "val_loss"]].dropna()

        plt.figure()
        plt.plot(trn_loss_df["epoch"], trn_loss_df["loss"], label="Training")
        plt.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Validation")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{log_dir}/loss.png")
        plt.close()

    if "acc" in metrics_df.columns and "val_acc" in metrics_df.columns:
        trn_acc_df = metrics_df[["epoch", "acc"]].dropna()
        val_acc_df = metrics_df[["epoch", "val_acc"]].dropna()

        plt.figure()
        plt.plot(trn_acc_df["epoch"], trn_acc_df["acc"], label="Training")
        plt.plot(val_acc_df["epoch"], val_acc_df["val_acc"], label="Validation")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{log_dir}/accuracy.png")
        plt.close()

    if "learning_rate" in metrics_df.columns:
        lr_df = metrics_df[["epoch", "learning_rate"]].dropna()
        plt.figure()
        plt.plot(lr_df["epoch"], lr_df["learning_rate"], marker="o", linestyle="-", label="Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.savefig(f"{log_dir}/learning_rate.png")
        plt.close()



def plot_pca(embeddings, labels=None, label_names=None, save_path=None):
    """
    Apply PCA to reduce embeddings to 2D and plot them with class-based colors.

    :param embeddings: NumPy array of shape (n_samples, n_features)
    :param labels: NumPy array of class labels (integers).
    :param label_names: Dictionary mapping class indices to class names.
    :param save_path: Optional file path to save the figure.
    """
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    # Check variance explained
    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance by first two components: {explained_var[0]:.2f}, {explained_var[1]:.2f}")

    # Define distinct colors for each class
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    # Scatter plot with class names
    for i, label in enumerate(unique_labels):
        idx = labels == label
        class_name = label_names.get(label, f"Class {label}") if label_names else f"Class {label}"
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1],
                    label=class_name, color=colors(i), s=50, alpha=0.7)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("2D PCA Projection of Embeddings")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_tsne(embeddings, labels=None, label_names=None, perplexity=30, save_path=None):
    """
    Apply t-SNE to reduce embeddings to 2D and plot them with specific class names.

    :param embeddings: NumPy array of shape (n_samples, n_features)
    :param labels: NumPy array of class labels (integers).
    :param label_names: Dictionary mapping class indices to class names.
    :param perplexity: t-SNE perplexity parameter.
    :param save_path: Optional file path to save the figure.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    # Define distinct colors for each class
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  # 'tab10' gives 10 distinguishable colors

    # Scatter plot with custom class names
    for i, label in enumerate(unique_labels):
        idx = labels == label
        class_name = label_names.get(label, f"Class {label}") if label_names else f"Class {label}"
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
                    label=class_name, color=colors(i), s=50, alpha=0.7)

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"2D t-SNE Projection (perplexity={perplexity})")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside for clarity

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    print(f"Classes {classes}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(classes.keys()))

    # Normalize by row (true labels) to get percentage (0 to 1)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert dictionary to list of class names
    class_labels = list(classes.values())

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")  # Show values as decimals (percentages)

    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_silhouette_analysis(embeddings, save_path, range_n_clusters=[8]):
    """
    Performs silhouette analysis and clustering visualization for given embeddings.

    Parameters:
        embeddings (numpy.ndarray): The embeddings to cluster, shape (num_samples, num_features).
        range_n_clusters (list): List of cluster numbers to evaluate.
    """
    X = np.array(embeddings)  # Ensure it's a NumPy array

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is: {silhouette_avg}")

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("Silhouette plot for various clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")

        ax2.set_title("Visualization of clustered data")
        ax2.set_xlabel("Feature space for 1st feature")
        ax2.set_ylabel("Feature space for 2nd feature")

        plt.suptitle(f"Silhouette analysis for KMeans clustering with n_clusters = {n_clusters}",
                     fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()