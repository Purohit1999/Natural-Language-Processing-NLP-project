import nltk, spacy
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

porter = PorterStemmer()
lancaster = LancasterStemmer()
wnl = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

sentences = [
    "Players were running and eating after the game.",
    "He goes to meetings and has eaten already.",
]

def compare(sentence: str):
    print("\nSENTENCE:", sentence)
    # token split (spaCy handles punctuation nicely)
    doc = nlp(sentence)
    tokens = [t.text for t in doc if not t.is_space]
    print("Tokens      :", tokens)
    print("Porter      :", [porter.stem(t) for t in tokens])
    print("Lancaster   :", [lancaster.stem(t) for t in tokens])
    print("WN Lemmas(v):", [wnl.lemmatize(t, pos="v") for t in tokens])
    print("spaCy Lemma :", [t.lemma_ for t in doc])

if __name__ == "__main__":
    for s in sentences:
        compare(s)
