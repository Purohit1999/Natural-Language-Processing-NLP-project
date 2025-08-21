# streamlit run text_cleaning_HW_8.py
import streamlit as st
import re, nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

st.title("Text Cleaning Demo")
txt = st.text_area("Paste text:", "The children are running to the playground!")

def clean(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP]
    return tokens

if st.button("Clean"):
    st.write(clean(txt))
