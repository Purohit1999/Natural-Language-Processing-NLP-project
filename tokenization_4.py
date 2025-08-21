import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import spacy
text = "I don't😊 know what to do."

# Python split
print(text.split())  

# NLTK
print(word_tokenize(text)) 

# Spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)

