# requirement_rewriter.py
# CLI:       python requirement_rewriter.py -t "I need to book a flight from Delhi to New York tomorrow."
# Streamlit: streamlit run requirement_rewriter.py

from __future__ import annotations
from typing import Dict, Optional
import argparse
import sys
import spacy

# --- spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    raise SystemExit("spaCy model not found. Install once:\n  python -m spacy download en_core_web_sm")

# --- CONFIG ---
GENERALIZE_DATES = True  # If any DATE/TIME is found, say "on a specific date"

DETERMINERS = {"a", "an", "the", "my", "your", "his", "her", "their", "our",
               "this", "that", "these", "those"}

def _add_article_if_missing(obj: Optional[str]) -> Optional[str]:
    """Add 'a/an' if object phrase lacks a determiner and looks singular."""
    if not obj:
        return obj
    words = obj.split()
    if not words:
        return obj
    first = words[0].lower()
    if first in DETERMINERS:
        return obj
    # crude plural check: if last word ends with 's', don't add article
    if words[-1].lower().endswith("s"):
        return obj
    article = "an" if words[0][0].lower() in "aeiou" else "a"
    return f"{article} {obj}"

# ---------- Core extraction ----------
def extract_slots(doc) -> Dict[str, Optional[str]]:
    """Extract action, object, from, to, location, when from a parsed Doc."""
    action = obj = from_loc = to_loc = location = when = None

    # 1) main action: ROOT verb or its xcomp
    root = [t for t in doc if t.dep_ == "ROOT"]
    if root:
        head = root[0]
        xcomp = next((c for c in head.children if c.dep_ == "xcomp" and c.pos_ == "VERB"), None)
        main = xcomp or (head if head.pos_ == "VERB" else None)
        if main:
            action = main.lemma_
        else:
            first_verb = next((t for t in doc if t.pos_ == "VERB"), None)
            action = first_verb.lemma_ if first_verb else None

    # 2) object candidates: dobj/attr/oprd around head/xcomp; fallback to noun chunks (not subject)
    candidates = []
    if root:
        head = root[0]
        candidates += [c for c in head.subtree if c.dep_ in {"dobj", "attr", "oprd"}]
        for c in head.children:
            if c.dep_ == "xcomp":
                candidates += [d for d in c.children if d.dep_ in {"dobj", "attr", "oprd"}]
    if not candidates:
        candidates = [nc.root for nc in doc.noun_chunks if nc.root.dep_ not in {"nsubj", "nsubjpass"}]

    def span_for_token(tok):
        for nc in doc.noun_chunks:
            if nc.start <= tok.i < nc.end:
                return nc
        return tok.subtree

    if candidates:
        best = max(candidates, key=lambda t: len(list(span_for_token(t))))
        span = span_for_token(best)
        obj = " ".join([t.text for t in span]) if hasattr(span, "__iter__") else str(span)
        if obj and obj.lower().startswith(("a ", "an ", "the ")):
            obj = " ".join(obj.split()[1:])
        obj = _add_article_if_missing(obj)

    # 3) prepositional phrases:
    #    - Keep dedicated 'from' and 'to'
    #    - Also capture a general 'location' for preps like in/at/inside/within
    LOCATION_PREPS = {"in", "at", "inside", "within", "into", "onto", "on"}  # 'on' sometimes for locations
    for token in doc:
        if token.dep_ == "prep":
            prep = token.text.lower()
            pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
            if not pobj:
                continue
            phrase = doc[token.i : (pobj.right_edge.i + 1)].text
            if prep == "from" and from_loc is None:
                from_loc = phrase
            elif prep == "to" and to_loc is None:
                to_loc = phrase
            elif prep in LOCATION_PREPS and location is None:
                location = phrase

    # 4) date/time entities
    ents = [e.text for e in doc.ents if e.label_ in {"DATE", "TIME"}]
    when = " ".join(ents) if ents else None

    return {
        "action": action,
        "object": obj,
        "from": from_loc,
        "to": to_loc,
        "location": location,
        "when": when,
    }

def _clean_when(when: Optional[str]) -> Optional[str]:
    if not when:
        return None
    if GENERALIZE_DATES:
        return "on a specific date"
    w = when.strip().lower()
    if w in {"today", "tomorrow", "tonight"}:
        return w
    return f"on {when}"

def render_requirement(slots: Dict[str, Optional[str]]) -> str:
    parts = []
    action, obj = slots.get("action"), slots.get("object")

    if action and obj:
        parts.append(f"The user wants to {action} {obj}")
    elif action:
        parts.append(f"The user wants to {action}")
    elif obj:
        parts.append(f"The user has a request about {obj}")
    else:
        return "The user's requirement could not be determined from the text."

    if slots.get("from"):
        parts.append(slots["from"])      # includes 'from ...'
    if slots.get("to"):
        parts.append(slots["to"])        # includes 'to ...'
    if slots.get("location"):
        parts.append(slots["location"])  # e.g., 'in Paris'

    when_phrase = _clean_when(slots.get("when"))
    if when_phrase:
        parts.append(when_phrase)

    out = " ".join(parts).strip()
    return out if out.endswith(".") else out + "."

def rewrite_requirement(text: str) -> str:
    doc = nlp(text)
    return render_requirement(extract_slots(doc))

# ---------- Streamlit app ----------
def app():
    import streamlit as st
    st.set_page_config(page_title="Requirement Rewriter", layout="centered")
    st.title("🧾 Requirement Rewriter (Action + Object)")

    text = st.text_area(
        "Enter user text:",
        "I want to reserve hotel room in Paris next week.",
        height=120
    )

    if st.button("Rewrite"):
        doc = nlp(text)
        st.subheader("Result")
        st.success(rewrite_requirement(text))

        st.subheader("Extracted slots")
        st.json(extract_slots(doc))
    else:
        st.info("Type a sentence and click **Rewrite**.")

# ---------- Entry points ----------
if __name__ == "__main__":
    if any(m.startswith("streamlit") for m in sys.modules):
        app()  # launched via: streamlit run requirement_rewriter.py
    else:
        parser = argparse.ArgumentParser(description="Extract action+object and rewrite as requirement.")
        parser.add_argument("-t", "--text", required=True, help="User text, e.g., 'I need to book a flight...'")
        args = parser.parse_args()
        print(rewrite_requirement(args.text))
