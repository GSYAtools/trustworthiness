import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

def plot_distance_distributions(name, A_emb, B_emb, output_dir="results"):
    def distances(x):
        x = np.array(x)
        centroid = x.mean(axis=0)
        return np.linalg.norm(x - centroid, axis=1)

    A_emb = np.array(A_emb)
    B_emb = np.array(B_emb)
    dist_A = distances(A_emb)
    dist_B = distances(B_emb)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(dist_A, fill=True, label="Prompt A", alpha=0.6)
    sns.kdeplot(dist_B, fill=True, label="Prompt B", alpha=0.6)
    plt.title(f"Distance to Centroid: {name}")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.legend()
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    plt.savefig(os.path.join(output_dir, name, "kde_distances.png"))
    plt.close()

def plot_projection(name, A_emb, B_emb, output_dir="results", method="tsne"):
    A_emb = np.array(A_emb)
    B_emb = np.array(B_emb)
    if A_emb.shape[1] == 1:
        print(f"[Info] Skipping {method.upper()} projection for binary embeddings in {name}")
        return

    X = np.concatenate([A_emb, B_emb])
    labels = ["A"] * len(A_emb) + ["B"] * len(B_emb)

    if method == "tsne":
        proj = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(X)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'.")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, alpha=0.7)
    plt.title(f"{method.upper()} Projection: {name}")
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    plt.savefig(os.path.join(output_dir, name, f"{method}_projection.png"))
    plt.close()

def plot_distributions(name, A_emb, B_emb, output_dir="results", disable=False):
    if disable:
        return
    plot_distance_distributions(name, A_emb, B_emb, output_dir)
    plot_projection(name, A_emb, B_emb, output_dir, method="tsne")
    plot_projection(name, A_emb, B_emb, output_dir, method="umap")