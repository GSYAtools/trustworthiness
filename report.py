import os
import json
import numpy as np

# Principios alineados con regulaciones como el EU AI Act
RELEVANT_FOR_LAWFULNESS = {
    "Justice": True,
    "Explicability": True,
    "Autonomy": True,
    "Technology": True,
    "Beneficence": False,
    "Non-maleficence": False
}

def compute_lawfulness_projection(metrics, thresholds):
    """
    Devuelve cumplimiento binario y continuo basado en umbrales por métrica.
    """
    violations = []
    penalties = []

    for m, val in metrics.items():
        threshold = thresholds.get(m)
        if threshold is not None:
            violations.append(val <= threshold)
            penalties.append(min(val / threshold, 1.0))  # Normalizado entre 0 y 1

    if not violations:
        raise ValueError("No thresholds matched the provided metrics. Cannot compute compliance.")

    binary = bool(all(violations))
    continuous = round(1.0 - sum(penalties) / len(penalties), 3)

    return {
        "binary": binary,
        "continuous": continuous
    }

def load_examples(prompt_name, output_dir="results"):
    """
    Carga examples.json si existe.
    """
    path = os.path.join(output_dir, prompt_name, "examples.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def to_native(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para evitar errores de serialización.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

def generate_report(prompt_name, config, metrics, thresholds, p_values=None, output_dir="results"):
    """
    Genera un archivo report.json que resume métricas, activación, cumplimiento normativo y p-valores si están disponibles.
    """
    if not metrics or not isinstance(metrics, dict):
        raise ValueError(f"Missing or malformed metrics for prompt {prompt_name}")

    if not thresholds or not isinstance(thresholds, dict):
        raise ValueError(f"Thresholds must be provided explicitly for prompt {prompt_name}")

    os.makedirs(os.path.join(output_dir, prompt_name), exist_ok=True)

    compliance = compute_lawfulness_projection(metrics, thresholds)
    principle = config.get("taxonomy_principle")
    is_regulatory = RELEVANT_FOR_LAWFULNESS.get(principle, False)

    report = {
        "name": prompt_name,
        "taxonomy_principle": principle,
        "evaluated_property": config.get("evaluated_property"),
        "metrics": metrics,
        "compliance": compliance,
        "lawfulness": {
            "is_regulatory": is_regulatory,
            "projected_binary": compliance["binary"] if is_regulatory else None,
            "projected_continuous": compliance["continuous"] if is_regulatory else None
        },
        "activated": {
            m: metrics[m] > thresholds.get(m, 1e9)
            for m in metrics if m in thresholds
        },
        "representations": [
            "kde_distances.png",
            "tsne_projection.png",
            "umap_projection.png"
        ]
    }

    # Añadir p-valores si se proporcionan
    if p_values:
        report["p_values"] = p_values

    # Añadir ejemplos si existen
    examples = load_examples(prompt_name, output_dir)
    if examples:
        report["examples"] = examples

    # Guardar reporte en formato JSON compatible
    out_path = os.path.join(output_dir, prompt_name, "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_native(report), f, indent=2, ensure_ascii=False)

    print(f" Report saved to {out_path}")