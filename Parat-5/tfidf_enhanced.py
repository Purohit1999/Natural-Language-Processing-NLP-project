# tfidf_enhanced.py
# pip install scikit-learn pandas matplotlib

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# --------- 1) Sample corpus (replace with your own) ----------
corpus = [
    "I love Python and AI.",
    "AI will shape the future.",
    "Python is great for data science and machine learning.",
    "I love building AI apps with Python!"
]

# --------- 2) Vectorizer with sensible defaults ----------
# tweak stop_words, ngram_range, min_df, max_df as needed
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",   # remove common English words
    ngram_range=(1, 2),     # unigrams + bigrams
    max_df=0.95,            # drop terms in >95% docs
    min_df=1,               # keep terms appearing in >=1 docs
    norm="l2"               # cosine-friendly vectors
)

X = vectorizer.fit_transform(corpus)              # sparse (n_docs x n_terms)
terms = np.array(vectorizer.get_feature_names_out())

# --------- 3) IDF table (sorted, easiest to read) ----------
idf = vectorizer.idf_
idf_df = pd.DataFrame({"term": terms, "idf": idf}).sort_values("idf", ascending=False)
print("\n=== IDF (higher = rarer/more discriminative) ===")
print(idf_df.to_string(index=False))

# --------- 4) TF-IDF matrix as a tidy DataFrame ----------
tfidf_df = pd.DataFrame(X.toarray(), columns=terms, index=[f"doc_{i+1}" for i in range(X.shape[0])])
print("\n=== TF-IDF matrix (rows=docs, cols=terms) ===")
print(tfidf_df.round(3).to_string())

# --------- 5) Top-k terms per document ----------
K = 5
print(f"\n=== Top {K} terms per document ===")
for i, row in enumerate(tfidf_df.values):
    top_idx = row.argsort()[::-1][:K]
    pairs = [f"{terms[j]}: {row[j]:.3f}" for j in top_idx if row[j] > 0]
    print(f"doc_{i+1} -> " + ", ".join(pairs))

# --------- 6) (Optional) save outputs ----------
SAVE = False  # set True to write CSVs
if SAVE:
    idf_df.to_csv("idf_values.csv", index=False)
    tfidf_df.to_csv("tfidf_matrix.csv")

# --------- 7) (Optional) quick bar chart of top terms of doc_1 ----------
PLOT = False  # set True to see a matplotlib chart
if PLOT:
    import matplotlib.pyplot as plt
    row = tfidf_df.iloc[0]
    top_idx = row.values.argsort()[::-1][:10]
    plt.figure(figsize=(10, 4))
    plt.bar(terms[top_idx], row.values[top_idx])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top TF-IDF terms for doc_1")
    plt.tight_layout()
    plt.show()
