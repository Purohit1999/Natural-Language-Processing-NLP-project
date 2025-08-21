import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("AI is transforming the world.")
tokens = [token.text for token in doc]
print(tokens)
