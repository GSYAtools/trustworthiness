import os
import json
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# Configuración
PROMPTS_DIR = "prompts"
OUTPUTS_DIR = "outputs"
RESULTS_DIR = "bootstrap_results"
N_BOOTSTRAP = 1000
TOP_N_PLOTS = 3

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_bootstrap_distribution(
    A_emb, B_emb, metric="w1", n_iter=1000,
    prompt_name="unknown", debug=False
):
    debug = True
    scores = []
    error_count = 0
    for i in range(n_iter):
        A_sample = A_emb[np.random.choice(len(A_emb), len(A_emb), replace=True)]
        B_sample = B_emb[np.random.choice(len(B_emb), len(B_emb), replace=True)]

        if np.isnan(A_sample).any() or np.isnan(B_sample).any():
            if debug and error_count < 5:
                print(f" NaN detected in embeddings for {prompt_name} [{metric}] at iteration {i}")
            error_count += 1
            continue

        try:
            if metric == "Wasserstein":
                cA = A_sample.mean(axis=0)
                cB = B_sample.mean(axis=0)
                proj_A = cdist(A_sample, cB.reshape(1, -1)).flatten()
                proj_B = cdist(B_sample, cA.reshape(1, -1)).flatten()
                
                if np.any(np.isnan(proj_A)) or np.any(np.isnan(proj_B)):
                    if error_count < 5:
                        print(f" NaN in projected distances for {prompt_name} [{metric}] at iter {i}")
                    error_count += 1
                    continue
                
                if np.allclose(proj_A, proj_A[0]) and np.allclose(proj_B, proj_B[0]):
                    if error_count < 5:
                        print(f" Degenerate projection for {prompt_name} [{metric}] (all distances identical)")
                    error_count += 1
                    continue
                
                score = wasserstein_distance(proj_A, proj_B)

            elif metric == "JS":
                bins = np.linspace(-1, 1, 50)
                hA, _ = np.histogram(A_sample @ A_sample.mean(axis=0), bins=bins, density=True)
                hB, _ = np.histogram(B_sample @ B_sample.mean(axis=0), bins=bins, density=True)

                hA += 1e-8
                hB += 1e-8
                hA /= hA.sum()
                hB /= hB.sum()

                if (np.count_nonzero(hA) <= 1 or np.count_nonzero(hB) <= 1) and error_count < 5:
                    print(f" Degenerate histogram for {prompt_name} [{metric}] at iteration {i} (sparse bins)")

                if np.any(np.isnan(hA)) or np.any(np.isnan(hB)):
                    if error_count < 5:
                        print(f" NaN in histogram for {prompt_name} [{metric}] at iteration {i}")
                    error_count += 1
                    continue

                score = jensenshannon(hA, hB)

            elif metric == "TV":
                bins = np.linspace(-1, 1, 50)
                hA, _ = np.histogram(A_sample @ A_sample.mean(axis=0), bins=bins, density=True)
                hB, _ = np.histogram(B_sample @ B_sample.mean(axis=0), bins=bins, density=True)

                hA += 1e-8
                hB += 1e-8
                hA /= hA.sum()
                hB /= hB.sum()

                score = 0.5 * np.sum(np.abs(hA - hB))

            else:
                if error_count < 5:
                    print(f" Unsupported metric: {metric}")
                error_count += 1
                continue

            if np.isnan(score) or np.isinf(score):
                if error_count < 5:
                    print(f" Invalid divergence value: {score} in {prompt_name} [{metric}] (iter {i})")
                error_count += 1
                continue

            scores.append(score)

        except Exception as e:
            if error_count < 5:
                print(f" Error in {prompt_name} [{metric}] at iteration {i}: {e}")
            error_count += 1
            continue

    scores = np.array(scores)
    if len(scores) == 0:
        print(f" No valid scores generated for {prompt_name} [{metric}]")
        return {
            "distribution": [],
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan")
        }

    return {
        "distribution": scores.tolist(),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }
# Análisis para todos los casos
summary = []
for fname in os.listdir(PROMPTS_DIR):
    if not fname.endswith(".json"):
        continue

    prompt_name = fname.replace(".json", "")
    config_path = os.path.join(PROMPTS_DIR, fname)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        subdir = os.path.join(OUTPUTS_DIR, prompt_name)
        path_A = os.path.join(subdir, "A_emb.npy")
        path_B = os.path.join(subdir, "B_emb.npy")
        if not (os.path.exists(path_A) and os.path.exists(path_B)):
            print(f"  Skipping {prompt_name}: embeddings not found in {subdir}/")
            continue

        A_emb, B_emb = np.load(path_A), np.load(path_B)
        metrics = config.get("metrics", [])
        for metric in metrics:
            print(f"→ Processing: {prompt_name} [{metric}]")
            stats = compute_bootstrap_distribution(A_emb, B_emb, metric, N_BOOTSTRAP)
            summary.append({
                "prompt": prompt_name,
                "metric": metric,
                **stats
            })

    except Exception as e:
        print(f" Error processing {prompt_name}: {e}")

# Clasificación de varianza relativa
def classify_variability(mean, std):
    if mean == 0 or np.isnan(mean) or np.isnan(std):
        return "invalid"
    rel_var = std / mean
    if rel_var < 0.10:
        return "very low"
    elif rel_var < 0.25:
        return "low"
    elif rel_var < 0.50:
        return "moderate"
    elif rel_var < 0.80:
        return "high"
    else:
        return "very high"

# Enriquecer el resumen con rel_var y clasificación
for entry in summary:
    mean = entry["mean"]
    std = entry["std"]
    rel_var = std / mean if mean and not np.isnan(mean) and not np.isnan(std) else float("nan")
    entry["rel_var"] = rel_var
    entry["variability"] = classify_variability(mean, std)

# Ordenar por rel_var descendente
summary_sorted = sorted(summary, key=lambda x: (np.isnan(x["rel_var"]), -x["rel_var"] if not np.isnan(x["rel_var"]) else -1))

# Guardar en JSON extendido
with open(os.path.join(RESULTS_DIR, "bootstrap_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_sorted, f, indent=2)

# Generar gráficos para todos los casos
for item in summary_sorted:
    dist = np.array(item["distribution"])
    if len(dist) == 0 or np.isnan(item["mean"]):
        print(f" Skipping plot for {item['prompt']} [{item['metric']}] due to empty or invalid distribution.")
        continue

    plt.figure(figsize=(6, 4))
    plt.hist(dist, bins=30, color="skyblue", edgecolor="black", alpha=0.9)
    plt.axvline(item["ci_low"], color="red", linestyle="--", label="95% CI")
    plt.axvline(item["ci_high"], color="red", linestyle="--")
    plt.axvline(item["mean"], color="blue", linestyle="-", label=f"Mean: {item['mean']:.3f}")
    plt.title(f"{item['prompt']} [{item['metric'].upper()}]\nRelVar: {item['rel_var']:.2f} ({item['variability']})")
    plt.xlabel(f"{item['metric'].upper()} Divergence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"{item['prompt']}_{item['metric']}_bootstrap.png")
    plt.savefig(plot_path)
    plt.close()

print(f"\n Bootstrap analysis complete. Results saved in: {RESULTS_DIR}/bootstrap_summary.json")
print(" All plots saved as PNG.")


 
