import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Carga del modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

REPRESENTATION_MAP = {
    "sentence_embedding": "sentence",
    "explanation_embedding": "explanation",
    "binary_decision": "binary",
    "token_logits": "logits"
}

def load_outputs(prompt_name, output_dir="outputs"):
    """
    Carga archivos de salida A/B en formato JSON para un prompt dado.
    """
    path_a = os.path.join(output_dir, prompt_name, "A_outputs.json")
    path_b = os.path.join(output_dir, prompt_name, "B_outputs.json")

    with open(path_a, encoding="utf-8") as f:
        A = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        B = json.load(f)

    assert len(A) == len(B), f"Mismatched sample sizes: {len(A)} vs {len(B)}"
    if len(A) != 30:
        print(f" Warning: Expected 30 samples, got {len(A)} for {prompt_name}")

    return A, B

def embed_sentences(texts):
    """
    Genera embeddings normalizados a partir de una lista de textos.
    """
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def load_cached_embeddings(path_a, path_b):
    if os.path.exists(path_a) and os.path.exists(path_b):
        return np.load(path_a), np.load(path_b)
    return None, None

def save_embeddings(path_a, path_b, A_emb, B_emb):
    np.save(path_a, A_emb)
    np.save(path_b, B_emb)

def embed_outputs(prompt_name, config, output_dir="outputs"):
    """
    Embebe los outputs generados para un prompt en función del tipo de representación.
    Usa cacheo si los embeddings ya han sido generados.
    """
    representation = config.get("representation")
    if representation not in REPRESENTATION_MAP:
        raise ValueError(f"Unsupported representation type: {representation}")

    A_texts, B_texts = load_outputs(prompt_name, output_dir)

    emb_dir = os.path.join(output_dir, prompt_name)
    path_emb_a = os.path.join(emb_dir, "A_emb.npy")
    path_emb_b = os.path.join(emb_dir, "B_emb.npy")

    cached_A, cached_B = load_cached_embeddings(path_emb_a, path_emb_b)
    if cached_A is not None and cached_B is not None:
        return cached_A, cached_B

    if representation in ["sentence_embedding", "explanation_embedding"]:
        A_emb = embed_sentences(A_texts)
        B_emb = embed_sentences(B_texts)

    elif representation == "binary_decision":
        def classify_binary(text):
            lower = text.lower()
            if any(x in lower for x in ["no", "sorry", "cannot", "not able", "unable"]):
                return 0
            return 1
        A_emb = np.array([classify_binary(t) for t in A_texts]).reshape(-1, 1)
        B_emb = np.array([classify_binary(t) for t in B_texts]).reshape(-1, 1)

    elif representation == "token_logits":
        raise NotImplementedError("Token logits not supported without access to model internals or API logprobs.")

    else:
        raise ValueError(f"Unknown representation: {representation}")

    save_embeddings(path_emb_a, path_emb_b, A_emb, B_emb)
    return A_emb, B_emb