# embed_viz.py
# Visualize sentence embeddings (Word2Vec) with PCA and t-SNE.

from __future__ import annotations
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import gensim.downloader as api

print("Loading word2vec-google-news-300 (cached if already downloaded)…")
W2V = api.load("word2vec-google-news-300")
DIM = W2V.vector_size
print("Loaded. dim =", DIM)

# ---------- helpers ----------
STOP = {"the","a","an","and","or","but","if","to","of","in","on","at","for","is","are","was","were",
        "be","been","being","with","by","as","that","this","these","those","it","its","from"}
WORD_RE = re.compile(r"[A-Za-z]+")

def tokenize(s: str):
    return [w.lower() for w in WORD_RE.findall(s) if w.lower() not in STOP]

def sent_vector(s: str, pool="mean") -> np.ndarray:
    toks = tokenize(s)
    vecs = [W2V[t] for t in toks if t in W2V.key_to_index]
    if not vecs:
        return np.zeros(DIM, dtype=np.float32)
    V = np.vstack(vecs)
    if pool == "max":
        return V.max(axis=0)
    return V.mean(axis=0)

# ---------- your sentences ----------
sentences = [
    "Machine learning is amazing",
    "AI will shape the future",
    "Python is great for data science",
    "I love building AI apps with Python",
    "Cooking pasta at home is relaxing",
    "Traveling to new countries expands the mind",
]

# compute vectors
X = np.vstack([sent_vector(s, pool="mean") for s in sentences])

# PCA (fast, deterministic)
pca2 = PCA(n_components=2, random_state=42).fit_transform(X)

# t-SNE (nonlinear, nice separation; adjust perplexity if few/many points)
tsne2 = TSNE(n_components=2, perplexity=min(30, max(5, len(sentences)//2)),
             learning_rate="auto", init="pca", random_state=42).fit_transform(X)

# ---------- plot ----------
plt.figure(figsize=(11,5))

# PCA subplot
plt.subplot(1,2,1)
plt.scatter(pca2[:,0], pca2[:,1])
for i, s in enumerate(sentences):
    plt.annotate(f"{i+1}", (pca2[i,0], pca2[i,1]), fontsize=9, xytext=(3,3), textcoords="offset points")
plt.title("PCA (2D)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.2)

# t-SNE subplot
plt.subplot(1,2,2)
plt.scatter(tsne2[:,0], tsne2[:,1])
for i, s in enumerate(sentences):
    plt.annotate(f"{i+1}", (tsne2[i,0], tsne2[i,1]), fontsize=9, xytext=(3,3), textcoords="offset points")
plt.title("t-SNE (2D)")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.grid(True, alpha=0.2)

plt.suptitle("Sentence Embeddings (Word2Vec) — PCA vs t-SNE\nLabels are sentence indices (see console).", y=0.98)
plt.tight_layout()
plt.show()

# print index → sentence mapping
print("\nSentence Index Map:")
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")
