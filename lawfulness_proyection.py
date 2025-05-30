import os
import json

# Principios relevantes para cumplimiento normativo (AI Act)
RELEVANT_FOR_LAWFULNESS = {
    "Justice": True,
    "Explicability": True,
    "Autonomy": True,
    "Technology": True,
    "Beneficence": False,
    "Non-maleficence": False
}

def compute_lawfulness_projection_across_cases(results_dir="results"):
    """
    Lee todos los report.json en results/*/ y calcula una proyección agregada de cumplimiento normativo.
    Solo incluye principios considerados relevantes para AI Act.
    """
    binary_scores = []
    continuous_scores = []
    cases_considered = []

    for case_name in os.listdir(results_dir):
        report_path = os.path.join(results_dir, case_name, "report.json")
        if not os.path.exists(report_path):
            continue

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        principle = report.get("taxonomy_principle")
        if not RELEVANT_FOR_LAWFULNESS.get(principle, False):
            continue

        compliance = report.get("compliance", {})
        binary_scores.append(bool(compliance.get("binary", False)))
        continuous_scores.append(float(compliance.get("continuous", 0.0)))
        cases_considered.append(case_name)

    result = {
        "lawfulness_binary": int(all(binary_scores)),
        "lawfulness_continuous": round(sum(continuous_scores) / len(continuous_scores), 3) if continuous_scores else None,
        "n_cases_considered": len(binary_scores),
        "included_principles": [k for k, v in RELEVANT_FOR_LAWFULNESS.items() if v],
        "included_cases": cases_considered
    }

    output_path = os.path.join(results_dir, "lawfulness_projection.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f" Lawfulness projection saved to: {output_path}")
    return result

if __name__ == "__main__":
    compute_lawfulness_projection_across_cases()