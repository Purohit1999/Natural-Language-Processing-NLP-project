import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Only download if missing (safe to run every time)
try:
    nltk.data.find("tokenizers/punkt/english.pickle")
    nltk.data.find("tokenizers/punkt_tab/english/")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

text = "Hello there! Welcome to NLP."
print("word tokens:", word_tokenize(text))
print("sentences  :", sent_tokenize(text))
