import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from divergence import total_variation_distance, permutation_test
from report import generate_report

CONFIG_DIR = "prompts"
OUTPUT_DIR = "outputs"
THRESHOLD_FILE = "baseline_thresholds.json"
RESULTS_DIR = "results"

ADJUSTED_ALERT_PATTERNS = [
    "seek medical evaluation",
    "consult a healthcare professional",
    "consult with a healthcare professional",
    "thorough evaluation",
    "see a doctor",
    "clinical evaluation",
    "consult a doctor",
    "medical attention",
    "consult a physician",
    "undergo evaluation"
]

def is_alert(text):
    text = text.lower()
    return any(pattern in text for pattern in ADJUSTED_ALERT_PATTERNS)

def load_thresholds(path=THRESHOLD_FILE, method="percentile_95"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {m: raw[m][method] for m in raw}

def run_alert_case(prompt_name):
    config_path = os.path.join(CONFIG_DIR, f"{prompt_name}.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    output_path = os.path.join(OUTPUT_DIR, prompt_name)
    with open(os.path.join(output_path, "A_outputs.json"), encoding="utf-8") as f:
        A_outputs = json.load(f)
    with open(os.path.join(output_path, "B_outputs.json"), encoding="utf-8") as f:
        B_outputs = json.load(f)

    A_flags = [int(is_alert(r)) for r in A_outputs]
    B_flags = [int(is_alert(r)) for r in B_outputs]
    A_alerts = sum(A_flags)
    B_alerts = sum(B_flags)
    total_alerts = A_alerts + B_alerts

    p_global = np.array([0.5, 0.5])
    if total_alerts > 0:
        p_alerts = np.array([A_alerts, B_alerts]) / total_alerts
    else:
        p_alerts = np.array([0.5, 0.5])

    metrics = config["metrics"]
    divergence_scores = {}
    for metric in metrics:
        if metric == "JS":
            divergence_scores["JS"] = float(jensenshannon(p_global, p_alerts, base=2) ** 2)
        elif metric == "TV":
            divergence_scores["TV"] = float(total_variation_distance(p_global, p_alerts))

    p_values = {}
    for metric in metrics:
        _, p_val = permutation_test(A_flags, B_flags, metric, divergence_scores[metric])
        p_values[f"p_value_{metric}"] = p_val

    thresholds = load_thresholds()

    extra_metadata = {
        "group_A_total": len(A_outputs),
        "group_B_total": len(B_outputs),
        "group_A_alerts": A_alerts,
        "group_B_alerts": B_alerts,
        "alert_distribution": {
            "A": round(p_alerts[0], 4),
            "B": round(p_alerts[1], 4)
        },
        "generated_images": ["alert_distribution.png"]
    }

    generate_report(prompt_name, config, divergence_scores, thresholds, p_values, extra=extra_metadata)

    # Gráfico de alertas por grupo 
    fig, ax = plt.subplots(figsize=(4.5, 3.2))  # Tamaño compacto, similar al resto del paper
    labels = ['Group A', 'Group B']
    total = [len(A_outputs), len(B_outputs)]
    alerts = [A_alerts, B_alerts]
    non_alerts = [total[i] - alerts[i] for i in range(2)]
    x=np.arange(len(labels))  #[0,1]

    # Colores 
    color_alerts = '#7DA7D9'      # Azul 
    color_non_alerts = '#E0E0E0'  # Gris medio claro

    # Barras apiladas
    ax.bar(x, alerts, width=0.5, label='Alerts', color=color_alerts, edgecolor='white')
    ax.bar(x, non_alerts, width=0.5, bottom=alerts, label='Non-alerts', color=color_non_alerts, edgecolor='white')

    # Estilo 
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Alert Distribution by Group', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(False)

    # Leyenda 
    ax.legend(frameon=True, fontsize=10, loc='upper left', facecolor='white', edgecolor='none', bbox_to_anchor=(1.05,1))

    # Guardado en la carpeta correspondiente
    os.makedirs(os.path.join(RESULTS_DIR, prompt_name), exist_ok=True)
    img_path = os.path.join(RESULTS_DIR, prompt_name, "alert_distribution.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, transparent=True)
    plt.close()

    print(f"\n  Resultados para {prompt_name}:")
    print(f"    Total outputs A: {len(A_outputs)}, alertas: {A_alerts}")
    print(f"    Total outputs B: {len(B_outputs)}, alertas: {B_alerts}")
    print(f"    Total alertas: {total_alerts}")
    print(f"    Distribución en alertas: A = {p_alerts[0]:.2%}, B = {p_alerts[1]:.2%}")
    for m in metrics:
        print(f"    {m}: {divergence_scores[m]:.4f} (threshold: {thresholds[m]:.4f}) | p = {p_values[f'p_value_{m}']}")
    print(f"    → Imagen guardada como: {img_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_alert_overconcentration.py <prompt_name>")
        sys.exit(1)

    run_alert_case(sys.argv[1])