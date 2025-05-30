import openai
import os
import json
import time
from dotenv import load_dotenv

# Cargar API key desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("Falta la variable OPENAI_API_KEY en .env")

# Parámetros del modelo y generación
MODEL_NAME = "gpt-4o"
N_SAMPLES = 30
SLEEP_TIME = 1.2
MAX_TOKENS = 75
TEMPERATURE = 0.7
TOP_P = 0.9

def call_openai(prompt, model=MODEL_NAME, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def generate_completions(prompt, n=N_SAMPLES, sleep_time=SLEEP_TIME):
    outputs = []
    for i in range(n):
        try:
            out = call_openai(prompt)
            if not out:
                print(f"[Warning] Respuesta vacía en iteración {i}")
                continue
            outputs.append(out)
            time.sleep(sleep_time)  # evita rate limit
        except Exception as e:
            print(f"[Error] Iteración {i} falló: {e}")
            continue
    return outputs


def save_if_not_exists(path, completions):
    """
    Guarda las completions si el archivo aún no existe.
    """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=2, ensure_ascii=False)


def generate_if_missing(prompt_config, output_dir="outputs"):
    """
    Genera los outputs A/B si no existen para un prompt.
    """
    name = prompt_config["name"]
    prompt_A = prompt_config["prompt_A"]
    prompt_B = prompt_config["prompt_B"]
    target_path = os.path.join(output_dir, name)
    os.makedirs(target_path, exist_ok=True)

    path_A = os.path.join(target_path, "A_outputs.json")
    path_B = os.path.join(target_path, "B_outputs.json")

    if not os.path.exists(path_A):
        print(f"[Sampling] Generando outputs para A: {name}")
        A = generate_completions(prompt_A)
        save_if_not_exists(path_A, A)
        if len(A) < N_SAMPLES:
            print(f"[Warning] Solo se generaron {len(A)} muestras para A ({name})")
    else:
        print(f"[Sampling] Outputs A ya existen: {name}")

    if not os.path.exists(path_B):
        print(f"[Sampling] Generando outputs para B: {name}")
        B = generate_completions(prompt_B)
        save_if_not_exists(path_B, B)
        if len(B) < N_SAMPLES:
            print(f"[Warning] Solo se generaron {len(B)} muestras para B ({name})")
    else:
        print(f"[Sampling] Outputs B ya existen: {name}")