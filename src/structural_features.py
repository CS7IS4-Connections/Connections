"""
structural_features.py
----------------------
Extract structural NLP features from a preprocessing CSV (Task 2):
spaCy POS proportions, character length, token length, TTR (type–token ratio
on alphabetic tokens), and maximum dependency depth (per sentence, then max).

Usage:
    python -m spacy download en_core_web_sm   # once per environment
    python src/structural_features.py \\
        --input data/samples/sample_5k.csv \\
        --output results/csv/sample_5k_structural.csv

    python src/structural_features.py --input data/processed/full.csv \\
        --output results/csv/full_structural.csv --batch-size 100
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# Universal POS tags emitted as proportions (stable column set for downstream stats).
POS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
)


def _token_depth(token) -> int:
    d = 0
    t = token
    # spaCy's Token.head returns Token objects; identity (`is`) is not reliable for
    # detecting the root because it can yield a different object instance.
    # Stop when the head token is the token itself (by index in the Doc).
    while t.head.i != t.i:
        d += 1
        t = t.head
    return d


def _doc_max_dep_depth(doc) -> int:
    depths: list[int] = []
    for sent in doc.sents:
        for t in sent:
            depths.append(_token_depth(t))
    return max(depths) if depths else 0


def _pos_proportions(doc) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for t in doc:
        if t.is_space:
            continue
        counts[t.pos_] += 1
    total = sum(counts.values())
    out: dict[str, float] = {}
    for tag in POS_TAGS:
        key = f"pos_{tag}"
        if total == 0:
            out[key] = np.nan
        else:
            out[key] = counts.get(tag, 0) / total
    return out


def _features_for_doc(doc) -> dict[str, float | int]:
    char_len = len(doc.text)
    tokens = [t for t in doc if not t.is_space]
    token_len = len(tokens)
    alpha_lower = [t.text.lower() for t in doc if t.is_alpha]
    if len(alpha_lower) == 0:
        ttr = np.nan
    else:
        ttr = len(set(alpha_lower)) / len(alpha_lower)

    pos = _pos_proportions(doc)
    dep_max = _doc_max_dep_depth(doc)

    row: dict[str, float | int] = {
        "char_len": char_len,
        "token_len": token_len,
        "ttr": ttr,
        "dep_depth_max": dep_max,
    }
    row.update(pos)
    return row


def _prefix_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def extract_column_features(nlp, texts: list[str], col_prefix: str, batch_size: int) -> pd.DataFrame:
    rows: list[dict] = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size),
        total=len(texts),
        desc=f"spaCy [{col_prefix}]",
        leave=False,
    ):
        feats = _features_for_doc(doc)
        rows.append(_prefix_keys(feats, col_prefix))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural features (spaCy) for preprocessing CSV")
    parser.add_argument("--input", required=True, help="Preprocessing CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path (original columns + features)")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy pipeline name")
    parser.add_argument("--batch-size", type=int, default=64, help="nlp.pipe batch size")
    parser.add_argument(
        "--text-cols",
        default="caption,article_lead",
        help="Comma-separated text column names to analyse",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug)")
    args = parser.parse_args()

    try:
        import spacy
    except ImportError:
        print("Install spaCy: pip install spacy", file=sys.stderr)
        sys.exit(1)

    try:
        nlp = spacy.load(args.model, disable=["ner"])
    except OSError:
        print(
            f"Model {args.model!r} not found. Run: python -m spacy download {args.model}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        print(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    feature_blocks: list[pd.DataFrame] = []
    for col in text_cols:
        series = df[col].fillna("").astype(str)
        texts = series.tolist()
        block = extract_column_features(nlp, texts, col, args.batch_size)
        feature_blocks.append(block)

    out = pd.concat([df.reset_index(drop=True)] + feature_blocks, axis=1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
