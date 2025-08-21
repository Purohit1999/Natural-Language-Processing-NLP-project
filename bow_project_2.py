from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

docs = [
    "The children are running to the playground",
    "The child runs every day to the school playground",
    "Playground games are fun for children",
]

pipe = Pipeline([
    ("cv", CountVectorizer(stop_words="english"))
])

X = pipe.fit_transform(docs)
vocab = pipe.named_steps["cv"].get_feature_names_out()
counts = np.asarray(X.sum(axis=0)).ravel()

top_idx = counts.argsort()[::-1]
print("Top terms by count:")
for i in top_idx[:10]:
    print(f"{vocab[i]:<15} {counts[i]}")
