# streamlit run stem_lemma_app.py
from __future__ import annotations
import re
import io
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# NLP
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud

import spacy
from spacy.lang.en import English

# ---------- one-time downloads (safe to call repeatedly) ----------
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------- load spaCy model ----------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # fall back to a blank English pipeline if model missing
    nlp = English()
    nlp.add_pipe("tok2vec")
    nlp.add_pipe("tagger")
    nlp.add_pipe("lemmatizer")
    nlp.add_pipe("attribute_ruler")

# ---------- helpers ----------
porter = PorterStemmer()
lancaster = LancasterStemmer()
wnl = WordNetLemmatizer()
STOP = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    # light normalization only (lower + keep apostrophes/numbers/letters)
    text = text.strip()
    return text

def tokenize(text: str, remove_stop=True, remove_punct=True, to_lower=True):
    doc = nlp(text)
    rows = []
    for t in doc:
        if remove_punct and (t.is_punct or t.is_space):
            continue
        tok = t.text
        if to_lower:
            tok = tok.lower()
        if remove_stop and tok in STOP:
            continue
        rows.append((tok, t.pos_, t.lemma_))
    return rows  # [(token, pos, spacy_lemma)]

def build_table(tokens):
    """
    tokens: list of (token_text, pos, spacy_lemma)
    returns DataFrame with Original, POS, Porter, Lancaster, Lemma(NLTK), Lemma(spaCy), flags
    """
    data = []
    for tok, pos, sp_lemma in tokens:
        porter_s = porter.stem(tok)
        lanc_s = lancaster.stem(tok)
        # map spaCy coarse POS to WordNet POS for better NLTK lemmatizer results
        wn_pos = {
            "NOUN": "n",
            "VERB": "v",
            "ADJ": "a",
            "ADV": "r"
        }.get(pos, "n")
        wn_lemma = wnl.lemmatize(tok, pos=wn_pos)

        stem_vs_lemma_diff = (porter_s != wn_lemma) or (lanc_s != wn_lemma)
        orig_vs_lemma_diff = tok != wn_lemma

        data.append({
            "Original": tok,
            "POS": pos,
            "Porter Stem": porter_s,
            "Lancaster Stem": lanc_s,
            "NLTK Lemma": wn_lemma,
            "spaCy Lemma": sp_lemma,
            "Stem≠Lemma": "❌" if stem_vs_lemma_diff else "✅",
            "Original≠Lemma": "❌" if orig_vs_lemma_diff else "✅",
        })
    return pd.DataFrame(data)

def generate_wordcloud(freqs: dict[str, int], title: str):
    if not freqs:
        st.info("Not enough tokens to build a word cloud.")
        return
    wc = WordCloud(width=900, height=450, background_color="white")
    wc = wc.generate_from_frequencies(freqs)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(fig)

# ---------- UI ----------
st.set_page_config(page_title="Stem vs Lemma (NLP 101)", layout="wide")
st.title("🧠 NLP 101: Stem vs Lemma — Comparison & WordCloud")

with st.sidebar:
    st.header("⚙️ Options")
    remove_stop = st.checkbox("Remove stopwords", value=True)
    remove_punct = st.checkbox("Remove punctuation", value=True)
    to_lower = st.checkbox("Lowercase tokens", value=True)

    st.markdown("---")
    cloud_mode = st.radio(
        "WordCloud Root Type",
        ["spaCy Lemmas", "NLTK Lemmas", "Porter Stems", "Lancaster Stems"],
        index=0,
    )

    st.markdown("---")
    st.caption("Tip: Paste a short paragraph to see differences clearly.")

# --------- Text input ---------
sample = (
    "Players were running and eating after the game. "
    "He goes to meetings and has eaten already. "
    "The children are running to the playground."
)

text = st.text_area("Enter a paragraph:", sample, height=140)
text = clean_text(text)

if st.button("Analyze"):
    tokens = tokenize(text, remove_stop=remove_stop, remove_punct=remove_punct, to_lower=to_lower)

    if not tokens:
        st.warning("No tokens after filtering. Try disabling some options or changing the text.")
        st.stop()

    df = build_table(tokens)

    st.subheader("🔎 Word Comparison Table (with POS)")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # download button for CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv,
        file_name="stem_vs_lemma.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------- WordCloud section ---------
    st.subheader("☁️ Root Word WordCloud")

    if cloud_mode == "spaCy Lemmas":
        roots = [r["spaCy Lemma"] if isinstance(r, dict) else r[2] for r in df.to_dict("records")]
        title = "WordCloud • spaCy Lemmas"
    elif cloud_mode == "NLTK Lemmas":
        roots = [r["NLTK Lemma"] for r in df.to_dict("records")]
        title = "WordCloud • NLTK Lemmas"
    elif cloud_mode == "Porter Stems":
        roots = [r["Porter Stem"] for r in df.to_dict("records")]
        title = "WordCloud • Porter Stems"
    else:
        roots = [r["Lancaster Stem"] for r in df.to_dict("records")]
        title = "WordCloud • Lancaster Stems"

    # remove tiny tokens (1-char) for nicer clouds
    roots = [r for r in roots if isinstance(r, str) and len(r) > 1]
    freqs = dict(Counter(roots))
    generate_wordcloud(freqs, title)

else:
    st.info("Paste text and click **Analyze** to see tokens, stems, lemmas, POS and the WordCloud.")
