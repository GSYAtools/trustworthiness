import os
import json
import sys

from embedder import embed_outputs
from divergence import compute_divergence, permutation_test
from report import generate_report
from visualize import plot_distributions

# Paths
CONFIG_DIR = "prompts"
OUTPUT_DIR = "outputs"
THRESHOLD_FILE = "baseline_thresholds.json"

def load_thresholds(path=THRESHOLD_FILE, method="percentile_95"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {m: raw[m][method] for m in raw}

def run_case(prompt_name):
    config_path = os.path.join(CONFIG_DIR, f"{prompt_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n Ejecutando evaluación para: {prompt_name}")
    print(f" → Taxonomy principle: {config.get('taxonomy_principle')}")
    print(f" → Evaluated property: {config.get('evaluated_property')}")

    # Embeddings
    A_emb, B_emb = embed_outputs(prompt_name, config, output_dir=OUTPUT_DIR)

    # Divergencia observada
    metrics = config["metrics"]
    divergence_scores = compute_divergence(A_emb, B_emb, metrics)

    # Test de hipótesis por permutación
    p_values = {}
    for metric in metrics:
        observed = divergence_scores[metric]
        _, pval = permutation_test(A_emb, B_emb, metric, observed)
        p_values[f"p_value_{metric}"] = pval

    # Cargar umbrales de baseline
    thresholds = load_thresholds()

    # Reporte y visualización
    generate_report(prompt_name, config, divergence_scores, thresholds=thresholds, p_values=p_values)
    plot_distributions(prompt_name, A_emb, B_emb, output_dir="results")

    print(f" Resultados guardados en results/{prompt_name}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_case.py <prompt_name>")
        sys.exit(1)

    prompt_name = sys.argv[1]
    run_case(prompt_name)