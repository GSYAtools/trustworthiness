import os
import json
import sys
from sampler import generate_if_missing

PROMPT_DIR = "prompts"
REQUIRED_KEYS = ["name", "prompt_A", "prompt_B", "representation"]

def validate_config(config, filename):
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Prompt config {filename} is missing keys: {missing}")

def prepare_all_prompts(filter_substr=None):
    prompt_files = [f for f in os.listdir(PROMPT_DIR) if f.endswith(".json")]
    
    if filter_substr:
        prompt_files = [f for f in prompt_files if filter_substr in f]

    for filename in prompt_files:
        path = os.path.join(PROMPT_DIR, filename)
        with open(path, encoding="utf-8") as f:
            config = json.load(f)

        validate_config(config, filename)
        print(f"==> Procesando: {filename}")
        generate_if_missing(config)

if __name__ == "__main__":
    # Permite pasar un filtro opcional por nombre
    filtro = sys.argv[1] if len(sys.argv) > 1 else None
    prepare_all_prompts(filter_substr=filtro)