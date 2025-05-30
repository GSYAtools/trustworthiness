import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, fisher_exact

# -----------------------------------
# Cálculo de distancias y histogramas
# -----------------------------------

def distances_to_centroid(embeddings):
    embeddings = np.array(embeddings)
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return distances

def histogram_distribution(distances, bins=30):
    hist, _ = np.histogram(distances, bins=bins, density=True)
    hist += 1e-8  # smoothing para evitar ceros
    hist /= hist.sum()
    return hist

def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

# -----------------------------------
# Cálculo principal de divergencia
# -----------------------------------

def compute_divergence(embeddings_a, embeddings_b, metrics: list, bins=30) -> dict:
    results = {}

    dist_a = distances_to_centroid(embeddings_a)
    dist_b = distances_to_centroid(embeddings_b)

    hist_a = histogram_distribution(dist_a, bins)
    hist_b = histogram_distribution(dist_b, bins)

    for metric in metrics:
        if metric == "JS":
            results["JS"] = float(jensenshannon(hist_a, hist_b, base=2))
        elif metric == "Wasserstein":
            results["Wasserstein"] = float(wasserstein_distance(dist_a, dist_b))
        elif metric == "TV":
            results["TV"] = float(total_variation_distance(hist_a, hist_b))
        else:
            raise ValueError(f"Unsupported divergence metric: {metric}")

    return results

# -----------------------------------
# Test exacto de Fisher para datos binarios
# -----------------------------------

def fisher_exact_test(embeddings_a, embeddings_b):
    """
    Aplica Fisher exact test si ambos conjuntos son listas 1D de valores 0 o 1.
    """
    a = np.array(embeddings_a).flatten()
    b = np.array(embeddings_b).flatten()

    if set(np.unique(a)).issubset({0, 1}) and set(np.unique(b)).issubset({0, 1}):
        # Tabla de contingencia: filas = grupos A y B, columnas = 0 y 1
        table = [
            [np.sum(a == 0), np.sum(a == 1)],
            [np.sum(b == 0), np.sum(b == 1)]
        ]
        _, p = fisher_exact(table, alternative='two-sided')
        return p
    else:
        return None  # No aplica

# -----------------------------------
# Test de permutación o Fisher según tipo
# -----------------------------------

def permutation_test(embeddings_a, embeddings_b, metric_name, observed, num_permutations=1000, bins=30):
    """
    Usa Fisher exact test si las entradas son binarias. En caso contrario, aplica permutación.
    """
    # Caso especial: binarios → usar Fisher
    p_fisher = fisher_exact_test(embeddings_a, embeddings_b)
    if p_fisher is not None:
        return observed, round(p_fisher, 5)

    # Permutación estándar
    combined = np.vstack([embeddings_a, embeddings_b])
    n = len(embeddings_a)
    count = 0

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n]
        perm_b = combined[n:]
        score = compute_divergence(perm_a, perm_b, [metric_name], bins=bins)[metric_name]
        if score >= observed:
            count += 1

    return observed, round(count / num_permutations, 5)