# ner_plus.py
from __future__ import annotations
import argparse, sys, webbrowser
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy
from spacy.tokens import Doc
from spacy import displacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy model 'en_core_web_sm' not found.")
    print("Install it once with:\n  python -m spacy download en_core_web_sm")
    raise

# -------- core helpers --------
def analyze(text: str) -> Doc:
    return nlp(text)

def ents_as_rows(doc: Doc) -> List[Tuple[str, str, int, int, str]]:
    rows = []
    for ent in doc.ents:
        sent = ent.sent.text.strip()
        rows.append((ent.text, ent.label_, ent.start_char, ent.end_char, sent))
    return rows

def print_table(rows: Iterable[Tuple[str, str, int, int, str]]) -> None:
    from textwrap import shorten
    print("\nEntities:")
    print(f"{'TEXT':30} {'LABEL':10} {'START':>6} {'END':>6}  SENTENCE")
    print("-"*90)
    for text, label, start, end, sent in rows:
        print(f"{text[:30]:30} {label:10} {start:6} {end:6}  {shorten(sent, width=60)}")

def print_label_counts(doc: Doc) -> None:
    from collections import Counter
    cnt = Counter([ent.label_ for ent in doc.ents])
    if not cnt:
        print("\nNo entities found.")
        return
    print("\nLabel frequencies:")
    for lab, c in cnt.most_common():
        print(f"  {lab:10} {c}")

def save_csv(rows, path: Path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "start_char", "end_char", "sentence"])
        w.writerows(rows)
    print(f"\n✅ Saved CSV → {path}")

def visualize(doc: Doc, open_browser: bool = True, out_html: Path | None = None) -> None:
    html = displacy.render(doc, style="ent", page=True)
    if out_html:
        out_html.write_text(html, encoding="utf-8")
        print(f"\n🖼  Wrote HTML visualization → {out_html}")
        if open_browser:
            webbrowser.open(out_html.as_uri())
    else:
        # fallback: start simple HTTP via displacy.serve (blocks)
        print("\nServing displacy at http://localhost:5000 (Ctrl+C to stop)")
        displacy.serve(doc, style="ent")

# -------- demo text (your Eiffel example, extended) --------
DEMO = (
    "The Eiffel Tower was built from 1887 to 1889 by French engineer Gustave Eiffel, "
    "whose company specialized in building metal frameworks and structures. "
    "Prices rose to €2,000 in 2020 according to the Paris tourism board."
)

# -------- CLI --------
def main():
    p = argparse.ArgumentParser(description="Quick spaCy NER utility")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-t", "--text", help="Analyze this text directly")
    g.add_argument("-f", "--file", type=Path, help="Analyze text from a file")
    p.add_argument("--csv", type=Path, help="Save entities to CSV at this path")
    p.add_argument("--viz", action="store_true", help="Open displacy visualization in browser")
    p.add_argument("--html", type=Path, help="Write displacy HTML to this path (non-blocking)")
    args = p.parse_args()

    if args.text:
        text = args.text
    elif args.file and args.file.exists():
        text = args.file.read_text(encoding="utf-8")
    else:
        text = DEMO
        print("(No input provided — using demo text.)")

    doc = analyze(text)
    rows = ents_as_rows(doc)

    print_table(rows)
    print_label_counts(doc)

    if args.csv:
        save_csv(rows, args.csv)

    if args.viz or args.html:
        visualize(doc, open_browser=True, out_html=args.html)

if __name__ == "__main__":
    main()
