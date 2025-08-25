# job_requirement_extractor.py
# CLI:       python job_requirement_extractor.py -t "Looking for a data science role in Bangalore with 2 years experience at Infosys."
# Streamlit: streamlit run job_requirement_extractor.py

from __future__ import annotations
import re
import sys
import argparse
from typing import Dict, Optional, Tuple

import spacy

# -------------------- spaCy model --------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    raise SystemExit("spaCy model not found. Install once:\n  python -m spacy download en_core_web_sm")

# -------------------- Helpers --------------------
DETERMINERS = {"a","an","the","my","your","his","her","their","our","this","that","these","those"}

# Common job-role cue words (extend as needed)
ROLE_KEYWORDS = {
    "data science","data scientist","data analyst","data analytics","machine learning engineer",
    "ml engineer","ml scientist","ai engineer","business analyst","software engineer",
    "backend developer","frontend developer","full stack developer","product manager",
    "data engineer","cloud engineer","devops","qa engineer","test engineer","bi developer",
    "power bi developer","tableau developer","financial analyst","accountant"
}

# number words -> digits for experience like "two years"
NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15
}

EXP_PATTERNS = [
    r"(?P<min>\d+)\s*[-–]\s*(?P<max>\d+)\s*(?:years?|yrs?)",      # 2-4 years
    r"(?P<num>\d+)\s*\+\s*(?:years?|yrs?)",                       # 3+ years
    r"(?P<num>\d+)\s*(?:years?|yrs?)",                            # 2 years
    r"(?P<word>{})\s*(?:years?|yrs?)".format("|".join(NUM_WORDS)) # two years
]

def _norm_spaces(s:str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _add_article_if_missing(phrase: Optional[str]) -> Optional[str]:
    if not phrase: return phrase
    words = phrase.split()
    if not words: return phrase
    if words[0].lower() in DETERMINERS: return phrase
    if words[-1].lower().endswith("s"): return phrase  # crude plural check
    article = "an" if words[0][0].lower() in "aeiou" else "a"
    return f"{article} {phrase}"

def _find_best_role(doc: spacy.tokens.Doc) -> Optional[str]:
    text_lower = doc.text.lower()
    # 1) direct keyword hit
    hits = [kw for kw in ROLE_KEYWORDS if kw in text_lower]
    if hits:
        # prefer longest match
        return max(hits, key=len)

    # 2) heuristic: noun chunks containing "engineer/analyst/scientist/developer/manager"
    head_terms = {"engineer","analyst","scientist","developer","manager","architect","consultant"}
    for nc in doc.noun_chunks:
        last = nc.root.text.lower()
        if last in head_terms or any(t.lemma_.lower() in head_terms for t in nc):
            # keep modifiers like "data", "machine learning"
            span = _norm_spaces(nc.text)
            return span.lower()
    return None

def _find_company(doc: spacy.tokens.Doc) -> Optional[str]:
    # Prefer ORG entities
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if orgs:
        # pick longest org name
        return max(orgs, key=len)
    # light fallback: after "at|with|for <ORG-like>"
    m = re.search(r"\b(?:at|with|for)\s+([A-Z][\w&.\- ]+)", doc.text)
    return m.group(1).strip() if m else None

def _find_location(doc: spacy.tokens.Doc) -> Optional[str]:
    # Prefer GPE/LOC entities
    locs = [ent.text for ent in doc.ents if ent.label_ in {"GPE","LOC"}]
    if locs:
        return max(locs, key=len)
    # prepositional phrases: in/at + proper noun region
    for token in doc:
        if token.dep_ == "prep" and token.text.lower() in {"in","at"}:
            pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
            if pobj:
                return _norm_spaces(doc[token.i: pobj.right_edge.i+1].text)  # "in Paris"
    return None

def _parse_experience(text:str) -> Optional[str]:
    t = text.lower()
    for pat in EXP_PATTERNS:
        m = re.search(pat, t)
        if not m: continue
        g = m.groupdict()
        if "min" in g and g["min"] and g.get("max"):
            return f"{g['min']}-{g['max']} years"
        if "num" in g and g["num"]:
            return f"{g['num']} years"
        if "word" in g and g["word"]:
            return f"{NUM_WORDS[g['word']]} years"
    return None

def extract_job_requirements(text: str) -> Dict[str, Optional[str]]:
    doc = nlp(text)
    role = _find_best_role(doc)
    company = _find_company(doc)
    location = _find_location(doc)
    experience = _parse_experience(text)
    # tidy role (title-case nicely for output)
    role_clean = role.title() if role else None
    # if location captured as "in Paris", keep just "Paris" for summary but store phrase too
    loc_for_sentence = None
    if location:
        loc_for_sentence = location.replace("in ", "").replace("at ", "")
    return {
        "role": role_clean,
        "location": loc_for_sentence,
        "experience": experience,
        "company": company
    }

def summarize_requirements(slots: Dict[str, Optional[str]]) -> str:
    parts = []
    role = slots.get("role")
    location = slots.get("location")
    experience = slots.get("experience")
    company = slots.get("company")

    if role and location:
        parts.append(f"The user wants a {role} role in {location}")
    elif role:
        parts.append(f"The user wants a {role} role")
    else:
        parts.append("The user is exploring job opportunities")

    if experience:
        parts.append(f"and has {experience} of experience")
    if company:
        parts.append(f"at {company}")

    sentence = _norm_spaces(" ".join(parts))
    return sentence if sentence.endswith(".") else sentence + "."

# -------------------- CLI entry --------------------
def _cli():
    parser = argparse.ArgumentParser(description="Extract job-related requirements and summarize them.")
    parser.add_argument("-t","--text", required=True, help="Free text describing job preference/experience")
    args = parser.parse_args()
    slots = extract_job_requirements(args.text)
    print(summarize_requirements(slots))

# -------------------- Streamlit app --------------------
def _app():
    import streamlit as st
    st.set_page_config(page_title="Job Requirement Extractor", layout="centered")
    st.title("🧳 Job Requirement Extractor")

    default = "I am looking for a data science role in Bangalore with 2 years experience at Infosys."
    text = st.text_area("Paste a sentence or paragraph:", default, height=120)

    if st.button("Extract & Summarize"):
        slots = extract_job_requirements(text)
        st.subheader("Summary")
        st.success(summarize_requirements(slots))

        st.subheader("Extracted fields")
        st.json(slots)
    else:
        st.info("Enter text and click **Extract & Summarize**.")

# -------------------- Main switch --------------------
if __name__ == "__main__":
    if any(m.startswith("streamlit") for m in sys.modules):
        _app()   # streamlit run job_requirement_extractor.py
    else:
        _cli()   # python job_requirement_extractor.py -t "..."
