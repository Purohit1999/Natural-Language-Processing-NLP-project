# text_cleaning_6.py — Streamlit "Clean My Text" with extra tasks
# Run: streamlit run text_cleaning_6.py
import re
import string
import streamlit as st

# ------------------------- NLTK STOPWORDS (lazy download) --------------------
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

# ------------------------------- HELPERS -------------------------------------
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U00002600-\U000026FF"   # misc symbols
    "]+", flags=re.UNICODE
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

def remove_punct_and_digits_str(s: str) -> str:
    return "".join(ch for ch in s if ch not in string.punctuation and not ch.isdigit())

def remove_punct_and_digits_regex(s: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", " ", s)

def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def remove_stopwords(tokens, stopset):
    return [t for t in tokens if t.lower() not in stopset]

def avg_word_len(tokens):
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)

# ------------------------------- UI ------------------------------------------
st.set_page_config(page_title="Clean My Text", layout="centered")
st.title("🧹 Clean My Text")

example = "Hi 😊! Email me at john.doe@example.com. Visit https://example.com 🚀 #NLP"
text = st.text_area("Enter text", example, height=160)

st.sidebar.header("Options")

# Base steps
do_lower = st.sidebar.checkbox("Lowercase", True)
method = st.sidebar.radio("Remove punctuation & numbers with…",
                          ("String methods", "Regex"), index=0)

# --- Task 2: Emoji handling ---
emoji_action = st.sidebar.selectbox(
    "Emoji handling (Task 2)",
    ["Do nothing", "Remove emojis", "Replace emojis with [emoji]"],
    index=0
)

# --- Task 3: Remove URLs / Emails ---
do_remove_urls = st.sidebar.checkbox("Remove URLs (Task 3)", True)
do_remove_emails = st.sidebar.checkbox("Remove Email addresses (Task 3)", True)

# --- Task 4: Custom stopwords input ---
do_stop = st.sidebar.checkbox("Remove stopwords (NLTK)", False)
custom_sw_str = st.sidebar.text_input(
    "Custom stopwords (comma-separated, Task 4)", placeholder="e.g., hello, please"
)

# Action
btn = st.sidebar.button("Clean Text")

# --------------------------- PIPELINE ----------------------------------------
def clean_pipeline(raw: str):
    steps = []
    original_tokens = raw.split()  # for Task 6 % removed baseline

    cur = raw
    steps.append(("Original", cur))

    # (Task 3) Remove URLs / Emails early
    if do_remove_urls:
        cur = URL_RE.sub(" ", cur)
        steps.append(("Removed URLs", cur))
    if do_remove_emails:
        cur = EMAIL_RE.sub(" ", cur)
        steps.append(("Removed Emails", cur))

    # (Task 2) Emoji handling
    if emoji_action == "Remove emojis":
        cur = EMOJI_RE.sub("", cur)
        steps.append(("Removed emojis", cur))
    elif emoji_action == "Replace emojis with [emoji]":
        cur = EMOJI_RE.sub(" [emoji] ", cur)
        steps.append(("Replaced emojis with [emoji]", cur))

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

    # (Task 4) Merge default and custom stopwords, then remove (if toggled)
    stopset = set()
    if do_stop and EN_STOPWORDS:
        stopset |= EN_STOPWORDS
    if custom_sw_str.strip():
        custom_set = {w.strip().lower() for w in custom_sw_str.split(",") if w.strip()}
        stopset |= custom_set

    removed_pct = None  # for Task 6
    if stopset:
        before = tokens[:]
        tokens = remove_stopwords(tokens, stopset)
        cur = " ".join(tokens)
        steps.append(("Removed stopwords (default/custom)", cur))
        # (Task 6) % stopwords removed
        if before:
            removed_pct = round(100 * (len(before) - len(tokens)) / len(before), 2)

    cleaned_text = " ".join(tokens) if tokens else cur
    return cleaned_text, steps, tokens, original_tokens, removed_pct

# --------------------------- RUN & DISPLAY -----------------------------------
if btn:
    cleaned, steps, tokens, original_tokens, removed_pct = clean_pipeline(text)

    st.subheader("✅ Cleaned Text")
    st.code(cleaned)

    st.subheader("🔎 Steps")
    for name, content in steps:
        with st.expander(name, expanded=False):
            st.write(content)

    # (Task 6) Metrics
    st.subheader("📊 Metrics (Task 6)")
    st.write(f"Average word length: **{avg_word_len(tokens):.2f}**")
    if removed_pct is not None:
        st.write(f"Stopwords removed: **{removed_pct}%**")

    st.subheader(f"🔤 Tokens ({len(tokens)})")
    st.write(tokens)

    st.download_button("⬇️ Download cleaned text", data=cleaned,
                       file_name="cleaned.txt", mime="text/plain")
else:
    st.info("Choose options on the left and click **Clean Text**.")
