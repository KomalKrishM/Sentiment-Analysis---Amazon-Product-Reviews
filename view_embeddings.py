import torch
import numpy as np
import pandas as pd
import sys

import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# Load embeddings from .pt file
def load_embeddings(path):
    embeddings = torch.load(path)

    # Convert to NumPy if it's a tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    return embeddings

def plot_umap(embeddings, labels, n_neighbors=15, min_dist=0.1, title="UMAP Projection", save_path=None):
    """
    embeddings: numpy array of shape (n_samples, n_features)
    labels: list or array of labels (can be int or str)
    """
    # Encode labels if they're not integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Run UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=encoded_labels, cmap='Spectral', alpha=0.6)
    plt.colorbar(scatter, ticks=range(len(np.unique(encoded_labels))))
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
    # plt.close()
    sys.exit(0)


def plot_tsne(embeddings, labels, perplexity=30, n_iter=1000, title="t-SNE Projection", save_path=None):
    """
    embeddings: numpy array of shape (n_samples, n_features)
    labels: list or array of labels (can be int or str)
    save_path: optional string path to save the figure (e.g., "tsne_plot.png")
    """
    # Encode labels (in case they're strings)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='pca', random_state=42)
    reduced = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=encoded_labels, cmap='Spectral', alpha=0.6)
    plt.colorbar(scatter, ticks=np.unique(encoded_labels))
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.grid(True)

    # Save the figure if specified
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

    sys.exit(0)


# Load the embeddings
embedding_path = "/Users/komalkrishnamogilipalepu/Downloads/archive/bert_test_embeddings.pt"  # update this with your actual file path
sentence_embeddings = load_embeddings(embedding_path)

print(sentence_embeddings.shape)

df = pd.read_csv("/Users/komalkrishnamogilipalepu/Downloads/archive/test_dataset.csv")
sentence_labels = df['label'].tolist()

print(len(sentence_labels))

# Load or define your labels
# For example: sentence_labels = [0, 1, 0, 1, ...]  (length = 25000)
# Make sure labels are the same length as number of embeddings

save_image_path = "/Users/komalkrishnamogilipalepu/Downloads/archive/UMAP_plot.png"

# # Now call the UMAP plot function
# plot_umap(sentence_embeddings, sentence_labels, title="UMAP of Loaded Embeddings", save_path=save_image_path)

# Randomly sample 10,000 points
N = 10000
indices = np.random.choice(sentence_embeddings.shape[0], N, replace=False)
sampled_embeddings = sentence_embeddings[indices]
sampled_labels = [sentence_labels[i] for i in indices]

# Now run t-SNE on sampled data
# plot_tsne(sampled_embeddings, sampled_labels, title="t-SNE on Sampled Data", save_path=save_image_path)

# Now call the UMAP plot function
plot_umap(sampled_embeddings, sampled_labels, title="UMAP of Loaded Embeddings", save_path=save_image_path)


