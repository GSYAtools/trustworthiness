import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from random import sample

def load_outputs(prompt_name, output_dir="outputs"):
    base = os.path.join(output_dir, prompt_name)
    with open(os.path.join(base, "A_outputs.json"), encoding="utf-8") as f:
        A_texts = json.load(f)
    with open(os.path.join(base, "B_outputs.json"), encoding="utf-8") as f:
        B_texts = json.load(f)
    return A_texts, B_texts

def load_embeddings(prompt_name, output_dir="outputs"):
    base = os.path.join(output_dir, prompt_name)
    A_emb = np.load(os.path.join(base, "A_emb.npy"))
    B_emb = np.load(os.path.join(base, "B_emb.npy"))
    return A_emb, B_emb

def most_different_pair(A_texts, B_texts, A_emb, B_emb):
    sim_matrix = cosine_similarity(A_emb, B_emb)
    a_idx, b_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    return {
        "similarity": float(sim_matrix[a_idx, b_idx]),
        "A_index": int(a_idx),
        "B_index": int(b_idx),
        "A_text": A_texts[a_idx],
        "B_text": B_texts[b_idx]
    }

def most_similar_pair(A_texts, B_texts, A_emb, B_emb):
    sim_matrix = cosine_similarity(A_emb, B_emb)
    a_idx, b_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    return {
        "similarity": float(sim_matrix[a_idx, b_idx]),
        "A_index": int(a_idx),
        "B_index": int(b_idx),
        "A_text": A_texts[a_idx],
        "B_text": B_texts[b_idx]
    }

def get_random_examples(A_texts, B_texts, n=3):
    return {
        "A_samples": sample(A_texts, min(n, len(A_texts))),
        "B_samples": sample(B_texts, min(n, len(B_texts)))
    }

def save_examples(prompt_name, examples, output_dir="results"):
    path = os.path.join(output_dir, prompt_name)
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, "examples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\n Saved examples to {out_path}")

def analyze_and_save(prompt_name):
    A_texts, B_texts = load_outputs(prompt_name)
    A_emb, B_emb = load_embeddings(prompt_name)

    examples = {}
    examples.update(get_random_examples(A_texts, B_texts))
    examples["most_similar"] = most_similar_pair(A_texts, B_texts, A_emb, B_emb)
    examples["most_different"] = most_different_pair(A_texts, B_texts, A_emb, B_emb)

    save_examples(prompt_name, examples)

    # Optional console preview
    print("\n Random A example:", examples["A_samples"][0])
    print(" Random B example:", examples["B_samples"][0])
    print("\n Most similar:\n", examples["most_similar"])
    print("\n Most different:\n", examples["most_different"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python analyze_outputs.py <prompt_name>")
        exit(1)

    prompt_name = sys.argv[1]
    analyze_and_save(prompt_name)