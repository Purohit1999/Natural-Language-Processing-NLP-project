# spaCy lemmatization
# NOTE (run once):  python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The children are running to the playground."
doc = nlp(text)

print("========== spaCy Lemmas ==========")
print([token.lemma_ for token in doc])
