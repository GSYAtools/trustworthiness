import csv

def update_summary_csv(prompt_name, config, metrics, thresholds, output_dir="results", summary_file="summary.csv"):
    compliance = compute_lawfulness_projection(metrics, thresholds)
    activated = {
        m: int(metrics[m] > thresholds.get(m, 1e9))
        for m in metrics if m in thresholds
    }

    row = {
        "prompt": prompt_name,
        "principle": config.get("taxonomy_principle"),
        "property": config.get("evaluated_property"),
        "compliance_score": compliance["continuous"],
        "compliance_binary": compliance["binary"]
    }

    for m in metrics:
        row[f"{m}_value"] = round(metrics[m], 4)
        row[f"{m}_activated"] = activated.get(m, "NA")

    csv_path = os.path.join(output_dir, summary_file)
    fieldnames = list(row.keys())

    # Si no existe, escribir encabezado
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f" Summary updated at {csv_path}")