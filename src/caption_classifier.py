"""
caption_classifier.py
---------------------
Rule-based caption-type classifier based on Marsh & White (2003).

Four types, applied in strict priority order (first match wins):
  Extractive   — caption reproduces article wording (direct quotes or lead overlap)
  Descriptive  — caption describes the image with deictic / visual-verb markers
  Expansive    — caption adds context beyond the image (temporal / causal / relative)
  Independent  — default; self-contained with minimal article connection

Classification rules use ONLY raw caption text and article_lead text.
No similarity scores (tfidf_sim, jaccard_sim, sbert_sim, entity_jaccard, …)
are referenced — those are dependent variables.

Usage:
    python src/caption_classifier.py --input results/entity.csv --output results/sample_results.csv
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Rule signal lists
# ---------------------------------------------------------------------------

DEICTIC_MARKERS = [
    "pictured", "shown", "seen here", "from left", "above", "below",
    "file photo", "file image", "left to right", "right to left",
    "poses for", "pose for", "stands in front", "at the scene",
    "at a press conference", "speaks to reporters", "speaks at",
]

VISUAL_VERBS = {
    "pose", "stand", "sit", "walk", "hold", "wave", "smile", "gesture",
}

TEMPORAL_MARKERS = [
    "yesterday", "last week", "last month", "last year", "in 2",
    "on monday", "on tuesday", "on wednesday", "on thursday",
    "on friday", "on saturday", "on sunday", "during", "following",
    "after", "amid", "despite", "since", "as part of", "in the wake of",
]

CAUSAL_CONNECTIVES = [
    "amid", "despite", "following", "after", "since", "as", "while",
    "although", "however",
]

RELATIVE_MARKERS = [", who ", ", which ", ", whose ", ", where "]

_QUOTE_RE = re.compile(r'["\u201c\u201d]')


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _has_quotes(caption: str) -> bool:
    return bool(_QUOTE_RE.search(caption))


def _has_deictic(text_lower: str) -> bool:
    return any(m in text_lower for m in DEICTIC_MARKERS)


def _has_visual_verb_open(text_lower: str, nlp) -> bool:
    """True if caption opens with a proper name followed by a visual verb."""
    doc = nlp(text_lower[:150])
    tokens = [t for t in doc if not t.is_space and t.is_alpha]
    if len(tokens) < 2:
        return False
    # "Name verb …" or "FirstName LastName verb …"
    if tokens[0].is_alpha and tokens[1].lemma_ in VISUAL_VERBS:
        return True
    if len(tokens) >= 3 and tokens[0].is_alpha and tokens[1].is_alpha and tokens[2].lemma_ in VISUAL_VERBS:
        return True
    return False


def _has_temporal(text_lower: str) -> bool:
    return any(m in text_lower for m in TEMPORAL_MARKERS)


def _has_causal(text_lower: str) -> bool:
    for m in CAUSAL_CONNECTIVES:
        if re.search(r"\b" + re.escape(m) + r"\b", text_lower):
            return True
    return False


def _has_relative(text_lower: str) -> bool:
    return any(m in text_lower for m in RELATIVE_MARKERS)


def _lead_overlap(caption: str, article_lead: str, nlp) -> bool:
    """True if >3 of the first 5 caption content lemmas appear in article lemmas."""
    cap_doc = nlp(caption[:400])
    art_doc = nlp(article_lead[:1200])

    cap_content = [
        t.lemma_.lower()
        for t in cap_doc
        if not t.is_stop and not t.is_punct and t.is_alpha
        and t.pos_ in {"NOUN", "VERB", "ADJ", "PROPN", "ADV"}
    ][:5]

    if not cap_content:
        return False

    art_lemmas = {
        t.lemma_.lower()
        for t in art_doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    }
    return sum(1 for w in cap_content if w in art_lemmas) > 3


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(caption: str, article_lead: str = "", nlp=None) -> str:
    """
    Classify a single caption.  Rules fire in priority order; first match wins.

    Parameters
    ----------
    caption      : raw caption text
    article_lead : raw article lead text (used only for Extractive lead-overlap check)
    nlp          : loaded spaCy model; if None the lead-overlap and visual-verb
                   checks are skipped (conservative fallback)
    """
    text_lower = caption.lower().strip()
    has_quotes  = _has_quotes(caption)
    has_deictic = _has_deictic(text_lower)

    # ── Rule 1: Extractive ──────────────────────────────────────────────────
    if has_quotes and not has_deictic:
        return "Extractive"
    if not has_deictic and nlp is not None and article_lead:
        if _lead_overlap(caption, article_lead, nlp):
            return "Extractive"

    # ── Rule 2: Descriptive ─────────────────────────────────────────────────
    if has_deictic and not has_quotes:
        return "Descriptive"
    if not has_quotes and nlp is not None:
        if _has_visual_verb_open(text_lower, nlp):
            return "Descriptive"

    # ── Rule 3: Expansive ───────────────────────────────────────────────────
    if not has_quotes and not has_deictic:
        if _has_temporal(text_lower) or _has_causal(text_lower) or _has_relative(text_lower):
            return "Expansive"

    # ── Rule 4: Independent (default) ───────────────────────────────────────
    return "Independent"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based caption classifier (Marsh & White 2003)")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--model",        default="en_core_web_sm",
                        help="spaCy model for lead-overlap and visual-verb checks")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--caption-col",  default="caption")
    parser.add_argument("--article-col",  default="article_lead")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    # Load spaCy — sm is sufficient for lemmatisation here
    try:
        import spacy
        nlp = spacy.load(args.model)
    except Exception as e:
        print(f"Warning: could not load spaCy model '{args.model}': {e}. "
              f"Proceeding without NLP signals (lead-overlap and visual-verb checks disabled).")
        nlp = None

    captions = df[args.caption_col].fillna("").astype(str).tolist()
    articles = (
        df[args.article_col].fillna("").astype(str).tolist()
        if args.article_col in df.columns
        else [""] * len(df)
    )

    labels = [
        classify(cap, art, nlp)
        for cap, art in tqdm(zip(captions, articles), total=len(captions),
                             desc="Classifying captions")
    ]
    df["caption_type"] = labels

    # ── Distribution report ─────────────────────────────────────────────────
    print("\n=== Caption Type Distribution ===")
    counts = df["caption_type"].value_counts()
    total  = len(df)
    for label, n in counts.items():
        print(f"  {label:<14s}  {n:>6,}  ({100 * n / total:.1f}%)")

    if "source" in df.columns:
        print("\nBy source:")
        pivot = (
            df.groupby(["source", "caption_type"])
              .size()
              .unstack(fill_value=0)
        )
        print(pivot.to_string())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
