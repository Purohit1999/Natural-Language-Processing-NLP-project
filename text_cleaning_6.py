# # text_cleaning_6.py
# # -------------------
# # Basic text-cleaning demo:
# # 1) lowercasing
# # 2) removing punctuation & numbers (string vs regex)
# # 3) removing stopwords (NLTK)
# # 4) simple regex-based cleanup pipeline

# import re
# import string

# # ---------- 1) Lowercasing ----------
# text = "Natural Language Processing"
# print("Lower Case :", text.lower(), " -> ", text.lower())

# # ---------- 2) Removing Punctuation & Numbers ----------
# text = "Hello!! I have 2 cats."
# # Using string methods
# cleaned_string = "".join(ch for ch in text if ch not in string.punctuation and not ch.isdigit())
# print("Using String Methods :", cleaned_string)  # -> "Hello I have  cats"

# # Using regex
# cleaned_regex = re.sub(r"[^a-zA-Z\s]", "", text)
# print("Regular Expression :", cleaned_regex)     # -> "Hello I have  cats"

# # ---------- 3) Removing Stopwords ----------
# # Stopwords are common words that often don't add meaning (e.g., "is", "the", "and")
# try:
#     from nltk.corpus import stopwords
#     from nltk import download as nltk_download
#     # make sure stopwords are present
#     try:
#         _ = stopwords.words("english")
#     except LookupError:
#         print("[nltk_data] Downloading package stopwords...")
#         nltk_download("stopwords")
#     sw = set(stopwords.words("english"))
#     sample_tokens = ["hello", "i", "have", "cats"]
#     without_sw = [w for w in sample_tokens if w.lower() not in sw]
#     print("After Removing Stop words :", without_sw)
# except Exception as e:
#     print("Skipping NLTK stopwords step (install nltk to enable). Error:", e)

# # ---------- 4) Regex-based text cleaning pipeline ----------
# def clean_text_pipeline(s: str) -> str:
#     s = s.lower()
#     # remove URLs
#     s = re.sub(r"https?://\S+|www\.\S+", " ", s)
#     # remove mentions/hashtags
#     s = re.sub(r"[@#]\w+", " ", s)
#     # keep only letters and spaces
#     s = re.sub(r"[^a-z\s]", " ", s)
#     # collapse extra spaces
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# raw = "Hello!! Visit https://example.com — I have 2 cats @john #pets"
# print("Regex-based Text Cleaning :", clean_text_pipeline(raw))

# app.py — Text Cleaning Playground (Streamlit)
# Run: streamlit run app.py

import re
import string
import streamlit as st

# ---- NLTK stopwords (download lazily) ---------------------------------------
@st.cache_resource
def load_stopwords():
    try:
        from nltk.corpus import stopwords
        from nltk import download as nltk_download
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk_download("stopwords")
        return set(stopwords.words("english"))
    except Exception:
        return set()

EN_STOPWORDS = load_stopwords()

# ---- Cleaning helpers --------------------------------------------------------
def remove_punct_and_digits_str(s: str) -> str:
    return "".join(ch for ch in s if ch not in string.punctuation and not ch.isdigit())

def remove_punct_and_digits_regex(s: str) -> str:
    # keep only letters and spaces
    return re.sub(r"[^a-zA-Z\s]", " ", s)

def remove_stopwords(tokens, stop_set):
    return [t for t in tokens if t.lower() not in stop_set]

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# ---- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Text Cleaning Playground", layout="centered")
st.title("🧹 Text Cleaning Playground")
st.caption("Lowercasing • Punctuation & Number removal • Stopwords • Regex pipeline")

example = "Hello!! I have 2 cats. Visit https://example.com @john #pets"
text = st.text_area("Enter text", example, height=160)

st.sidebar.header("Cleaning steps")
do_lower = st.sidebar.checkbox("1) Lowercase", True)

method = st.sidebar.radio(
    "2) Remove punctuation & numbers using…",
    ("String methods", "Regex"),
    index=0
)

do_stop = st.sidebar.checkbox("3) Remove stopwords (NLTK)", False)
stop_src = st.sidebar.selectbox("Stopword language", ["english"], index=0, disabled=not do_stop)

extra_regex = st.sidebar.checkbox("4) Extra regex cleanup (URLs, @mentions, #hashtags)", True)

st.sidebar.markdown("---")
btn = st.sidebar.button("Clean Text")

# ---- Processing --------------------------------------------------------------
def run_cleaning(txt: str):
    steps = []

    cur = txt
    steps.append(("Original", cur))

    if extra_regex:
        cur = re.sub(r"https?://\S+|www\.\S+", " ", cur)   # URLs
        cur = re.sub(r"[@#]\w+", " ", cur)                 # mentions/hashtags
        steps.append(("Remove URLs & mentions/hashtags", cur))

    if do_lower:
        cur = cur.lower()
        steps.append(("Lowercased", cur))

    if method == "String methods":
        cur = remove_punct_and_digits_str(cur)
        steps.append(("Removed punctuation & digits (string)", cur))
    else:
        cur = remove_punct_and_digits_regex(cur)
        steps.append(("Removed punctuation & digits (regex)", cur))

    cur = collapse_spaces(cur)
    steps.append(("Collapsed spaces", cur))

    tokens = cur.split()

    if do_stop:
        if EN_STOPWORDS:
            tokens = remove_stopwords(tokens, EN_STOPWORDS)
            cur = " ".join(tokens)
            steps.append(("Removed stopwords", cur))
        else:
            steps.append(("Stopwords skipped (NLTK not available)", cur))

    return cur, steps, tokens

if btn:
    cleaned, steps, tokens = run_cleaning(text)

    st.subheader("✅ Cleaned Text")
    st.code(cleaned)

    st.subheader("🔎 Steps")
    for name, content in steps:
        with st.expander(name, expanded=False):
            st.write(content)

    st.subheader(f"🔤 Tokens ({len(tokens)})")
    st.write(tokens)

    st.download_button(
        "⬇️ Download cleaned text",
        data=cleaned,
        file_name="cleaned.txt",
        mime="text/plain"
    )
else:
    st.info("Choose options on the left and click **Clean Text**.")
