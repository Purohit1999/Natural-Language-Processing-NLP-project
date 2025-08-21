# tokenization_5_Project.py
import streamlit as st
import pandas as pd
from collections import Counter
import string

st.set_page_config(page_title="Tokenization Homework", page_icon="📝")
st.title("📝 Tokenization Homework")

# ---------- CACHED SETUP ----------
@st.cache_resource
def ensure_nltk():
    """Ensure punkt, punkt_tab, stopwords; return tokenizers + stopword set."""
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords

    # punkt
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # punkt_tab (newer NLTK)
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # fine if unavailable

    # stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    return word_tokenize, sent_tokenize, set(stopwords.words("english"))

word_tokenize, sent_tokenize, STOP_WORDS = ensure_nltk()

# ---------- SMALL HELPERS ----------
def strip_punct(tokens):
    """Remove pure punctuation tokens (Task-3 requirement)."""
    return [t for t in tokens if t not in string.punctuation and any(ch.isalnum() for ch in t)]

def remove_stopwords(tokens):
    """Return tokens with stopwords removed + how many were removed (Task-2)."""
    cleaned = [t for t in tokens if t.lower() not in STOP_WORDS]
    return cleaned, len(tokens) - len(cleaned)

def top_k_counts(tokens, k=5):
    """Top-k word frequencies as a DataFrame (lowercased, punctuation already removed)."""
    counts = Counter([t.lower() for t in tokens])
    top = counts.most_common(k)
    return pd.DataFrame(top, columns=["token", "count"]).set_index("token")

# ---------- UI ----------
txt = st.text_area(
    "Enter text",
    placeholder="Paste a few sentences here…",
    height=160,
)

if st.button("Process"):
    if not txt.strip():
        st.warning("Please enter some text.")
        st.stop()

    # --- Tokenize (words & sentences)
    word_tokens = word_tokenize(txt)
    sent_tokens = sent_tokenize(txt)

    st.subheader("Task 1 — Sentence tokenization")
    st.write(f"**Number of sentences:** {len(sent_tokens)}")
    with st.expander("Show sentences"):
        for i, s in enumerate(sent_tokens, 1):
            st.write(f"{i}. {s}")

    # First & last word (ignore punctuation-only tokens)
    non_punct = strip_punct(word_tokens)
    if non_punct:
        st.write(f"**First word:** `{non_punct[0]}`")
        st.write(f"**Last word:** `{non_punct[-1]}`")
    else:
        st.info("No non-punctuation words found to display first/last.")

    st.divider()

    st.subheader("Task 2 — Remove English stopwords")
    cleaned_tokens, removed_count = remove_stopwords(non_punct)
    st.write(f"**Removed stopwords:** {removed_count}")
    cols = st.columns(2)
    with cols[0]:
        st.write("**Original tokens (punctuation removed):**")
        st.write(non_punct)
    with cols[1]:
        st.write("**Cleaned tokens (no stopwords):**")
        st.write(cleaned_tokens)

    st.divider()

    st.subheader("Task 3 — Top-5 most common words (punctuation excluded)")
    # As per the task: exclude punctuation only (do NOT remove stopwords here)
    top_df = top_k_counts(non_punct, k=5)
    st.bar_chart(top_df)

    with st.expander("See counts table"):
        st.dataframe(top_df)

    st.success("Done ✅")
