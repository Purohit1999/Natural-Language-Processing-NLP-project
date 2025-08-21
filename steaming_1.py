# Stemming demo
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer

nltk.download("punkt", quiet=True)  # harmless if already downloaded

words = [
    "eating", "eats", "eaten", "going", "gone", "goes",
    "fairly", "sportingly", "running", "better"
]

porter = PorterStemmer()
lancaster = LancasterStemmer()
# RegexpStemmer takes the regex as the first positional arg (no 'pattern=' kwarg)
regex = RegexpStemmer(r"(ing|ed|s)$", min=4)

print("========== Porter Stemmer ==========")
for w in words:
    print(f"{w:>12} -> {porter.stem(w)}")

print("\n========== Lancaster Stemmer ==========")
for w in words:
    print(f"{w:>12} -> {lancaster.stem(w)}")

print("\n========== Regexp Stemmer (ing|ed|s) ==========")
for w in words:
    print(f"{w:>12} -> {regex.stem(w)}")
