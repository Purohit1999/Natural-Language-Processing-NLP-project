# nltk_setup.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    return word_tokenize, sent_tokenize
