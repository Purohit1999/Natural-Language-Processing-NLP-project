# verify_nltk.py
import os, sys
from nltk.data import find

print("Python exe :", sys.executable)
print("NLTK_DATA  :", os.environ.get("NLTK_DATA"))
print("punkt      :", find("tokenizers/punkt/english.pickle"))
print("punkt_tab  :", find("tokenizers/punkt_tab/english/"))
