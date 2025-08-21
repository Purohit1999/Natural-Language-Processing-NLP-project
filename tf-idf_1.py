from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "The children are running to the playground",
    "The child runs every day to the school playground",
    "Playground games are fun for children",
]

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(docs)

print("Vocabulary:", tfidf.get_feature_names_out().tolist())
print("TF-IDF matrix shape:", X.shape)
print("First doc TF-IDF:", X[0].toarray())
