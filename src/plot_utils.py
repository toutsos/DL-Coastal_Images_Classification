import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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