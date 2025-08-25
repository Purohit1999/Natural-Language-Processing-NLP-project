# tiny_w2v_demo.py
# Train a Word2Vec model on a tiny dataset and find the top-3 words
# similar to "learning".

from gensim.models import Word2Vec
from pprint import pprint
import numpy as np
import random

# ----- 1) tiny toy corpus -----
data = [
    ["i", "love", "machine", "learning"],
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "is", "part", "of", "machine", "learning"],
]

# Reproducibility
np.random.seed(42)
random.seed(42)

# ----- 2) train Word2Vec -----
# vector_size=50 (embedding dims)
# window=2 (context window)
# min_count=1 (keep all tokens)
# sg=1 (skip-gram; use sg=0 for CBOW)
model = Word2Vec(
    sentences=data,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,
    workers=1,
    epochs=300,
    seed=42,
)

# ----- 3) explore vocab & similarities -----
print("Vocabulary:", list(model.wv.key_to_index.keys()))

print("\nTop-3 words similar to 'learning':")
pprint(model.wv.most_similar("learning", topn=3))

# (Optional) a few pairwise similarities
def sim(a, b):
    return float(model.wv.similarity(a, b))

print("\nSome cosine similarities:")
for a, b in [("learning", "machine"), ("learning", "deep"), ("learning", "fun"), ("machine", "deep")]:
    print(f"sim({a!r}, {b!r}) = {sim(a,b):.3f}")
