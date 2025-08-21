from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love natural language processing",
    "I love love data",
    "NLP loves me"
]

cv = CountVectorizer()
X = cv.fit_transform(corpus)

print("Vocabulary:", cv.get_feature_names_out().tolist())
print("BoW matrix:\n", X.toarray())
