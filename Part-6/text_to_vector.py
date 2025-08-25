# text_to_vector_enhanced.py
# Usage:
#   python text_to_vector_enhanced.py -t "Machine learning is amazing"
#   python text_to_vector_enhanced.py -f sentences.txt
# Options:
#   --pool mean|max|tfidf   (default: mean)

from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Optional

import numpy as np

# --- Try to load GoogleNews word2vec (300d) via gensim ---
try:
    import gensim.downloader as api
    print("Loading word2vec-google-news-300 (≈1.6 GB)…")
    W2V = api.load("word2vec-google-news-300")  # may take minutes first time
    DIM = W2V.vector_size
    print("Loaded embeddings with dim =", DIM)
except Exception as e:
    W2V, DIM = None, 300
    print("WARNING: Could not load 'word2vec-google-news-300'.",
          "Make sure you have internet/disk space.", f"Details: {e}")

# --- Basic text cleaning ---
STOP = {
    "the","a","an","and","or","but","if","to","of","in","on","at","for","is","are","was","were",
    "be","been","being","with","by","as","that","this","these","those","it","its","from"
}

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # words like don't -> dont

def tokenize(text: str) -> List[str]:
    words = [w.lower() for w in TOKEN_RE.findall(text)]
    return [w for w in words if w not in STOP and len(w) > 1]

def embed_tokens(tokens: List[str]) -> List[np.ndarray]:
    """Return list of vectors for tokens that exist in the model."""
    if W2V is None:
        return []
    vecs = []
    for t in tokens:
        if t in W2V.key_to_index:
            vecs.append(W2V[t])
    return vecs

# --- Pooling functions ---
def pool_mean(vecs: List[np.ndarray]) -> np.ndarray:
    return np.mean(vecs, axis=0) if vecs else np.zeros(DIM, dtype=np.float32)

def pool_max(vecs: List[np.ndarray]) -> np.ndarray:
    return np.max(vecs, axis=0) if vecs else np.zeros(DIM, dtype=np.float32)

def pool_tfidf(tokens: List[str], vecs: List[np.ndarray]) -> np.ndarray:
    """
    Simple TF-IDF weighting: idf ≈ log(N / df) using GoogleNews vocab counts if available,
    otherwise fall back to mean.
    """
    if not tokens or not vecs:
        return np.zeros(DIM, dtype=np.float32)
    if W2V is None or not hasattr(W2V, "get_vecattr"):
        return pool_mean(vecs)

    # build weights aligned to vecs list
    weights = []
    for t in tokens:
        if t in W2V.key_to_index:
            # `count` is available for some gensim models; if not, use 1
            try:
                # google-news model doesn’t expose raw counts; use rank heuristic
                rank = W2V.get_vecattr(t, "count")  # may raise
                df = max(rank, 1)
            except Exception:
                # heuristic: lower rank (frequent) -> smaller weight
                df = W2V.key_to_index.get(t, 3000000)
            idf = np.log(1 + 3000000 / (1 + df))
            weights.append(idf)
    if not weights:
        return pool_mean(vecs)

    weights = np.array(weights, dtype=np.float32)
    M = np.vstack(vecs)
    w = weights / (weights.sum() + 1e-9)
    return (M * w[:, None]).sum(axis=0)

def sentence_vector(
    text: str,
    pool: str = "mean"
) -> Tuple[np.ndarray, List[str], int]:
    tokens = tokenize(text)
    vecs = embed_tokens(tokens)
    oov = len(tokens) - len(vecs)

    if pool == "max":
        v = pool_max(vecs)
    elif pool == "tfidf":
        v = pool_tfidf([t for t in tokens if t in W2V.key_to_index], vecs)
    else:
        v = pool_mean(vecs)

    return v.astype(np.float32), tokens, oov

def process_lines(lines: List[str], pool: str = "mean") -> np.ndarray:
    out = []
    for i, line in enumerate(lines, 1):
        v, toks, oov = sentence_vector(line, pool=pool)
        print(f"[{i}] tokens={len(toks)}  OOV={oov}  vector_norm={np.linalg.norm(v):.4f}")
        out.append(v)
    return np.vstack(out) if out else np.zeros((0, DIM), dtype=np.float32)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Convert text to vectors using Word2Vec with pooled embeddings.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-t", "--text", help="Single sentence to vectorize.")
    src.add_argument("-f", "--file", help="Path to a text file (one sentence per line).")
    ap.add_argument("--pool", choices=["mean","max","tfidf"], default="mean", help="Pooling method.")
    ap.add_argument("--save", help="(Optional) Save vectors to .npy file.")
    args = ap.parse_args()

    if W2V is None:
        print("ERROR: Embedding model not available. Connect to internet and try again, "
              "or pre-download with gensim.downloader.")
        return

    if args.text:
        v, toks, oov = sentence_vector(args.text, pool=args.pool)
        print("\nTokens:", toks)
        print(f"OOV words: {oov}")
        print("Vector shape:", v.shape)
        print("Vector (first 8 dims):", np.round(v[:8], 6))
        if args.save:
            np.save(args.save, v)
            print(f"Saved to {args.save}.npy")
    else:
        with open(args.file, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        mat = process_lines(lines, pool=args.pool)
        print("\nMatrix shape:", mat.shape)
        if args.save:
            np.save(args.save, mat)
            print(f"Saved to {args.save}.npy")

if __name__ == "__main__":
    main()
