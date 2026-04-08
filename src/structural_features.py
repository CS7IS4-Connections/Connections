"""
structural_features.py
----------------------
Extract structural NLP features from captions in a preprocessing CSV.

All features are derived from the caption column only, except the four
POS-overlap columns which compare caption lemmas against article_lead lemmas.

Features added:
  Length     : token_count, word_count, char_count, sent_count
  Complexity : avg_sent_len, dep_depth, clause_count
  Vocabulary : ttr, content_word_prop, propn_prop
  POS overlap: noun_overlap, verb_overlap, adj_overlap, propn_overlap

Usage:
    python -m spacy download en_core_web_trf   # once per environment
    python src/structural_features.py \\
        --input  data/samples/sample_data_clean.csv \\
        --output results/structural.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

CLAUSAL_DEPS = {"ccomp", "xcomp", "advcl", "relcl", "acl"}
CONTENT_POS  = {"NOUN", "VERB", "ADJ", "ADV"}


def _token_depth(token) -> int:
    d, t = 0, token
    while t.head.i != t.i:
        d += 1
        t = t.head
    return d


def _doc_max_dep_depth(doc) -> int:
    depths = [_token_depth(t) for t in doc if not t.is_space]
    return max(depths) if depths else 0


def _pos_lemmas(doc, pos_tag: str) -> set[str]:
    return {t.lemma_.lower() for t in doc if t.pos_ == pos_tag and not t.is_space}


def _overlap(cap_set: set, art_set: set) -> float:
    if not cap_set:
        return 0.0
    return len(cap_set & art_set) / len(cap_set)


def extract_features(cap_doc, art_doc) -> dict:
    tokens = [t for t in cap_doc if not t.is_space]
    words  = [t for t in tokens if t.is_alpha]
    sents  = list(cap_doc.sents)

    token_count = len(tokens)
    word_count  = len(words)
    char_count  = len(cap_doc.text)
    sent_count  = len(sents)

    sent_lens   = [len([t for t in s if not t.is_space]) for s in sents]
    avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0
    dep_depth    = _doc_max_dep_depth(cap_doc)
    clause_count = 1 + sum(1 for t in tokens if t.dep_ in CLAUSAL_DEPS)

    alpha_lower      = [t.text.lower() for t in cap_doc if t.is_alpha]
    ttr              = len(set(alpha_lower)) / len(alpha_lower) if alpha_lower else np.nan
    content_word_prop = (
        sum(1 for t in tokens if t.pos_ in CONTENT_POS) / token_count
        if token_count else np.nan
    )
    propn_prop = (
        sum(1 for t in tokens if t.pos_ == "PROPN") / token_count
        if token_count else np.nan
    )

    noun_overlap  = _overlap(_pos_lemmas(cap_doc, "NOUN"),  _pos_lemmas(art_doc, "NOUN"))
    verb_overlap  = _overlap(_pos_lemmas(cap_doc, "VERB"),  _pos_lemmas(art_doc, "VERB"))
    adj_overlap   = _overlap(_pos_lemmas(cap_doc, "ADJ"),   _pos_lemmas(art_doc, "ADJ"))
    propn_overlap = _overlap(_pos_lemmas(cap_doc, "PROPN"), _pos_lemmas(art_doc, "PROPN"))

    return {
        "token_count":        token_count,
        "word_count":         word_count,
        "char_count":         char_count,
        "sent_count":         sent_count,
        "avg_sent_len":       round(avg_sent_len, 4),
        "dep_depth":          dep_depth,
        "clause_count":       clause_count,
        "ttr":                ttr,
        "content_word_prop":  content_word_prop,
        "propn_prop":         propn_prop,
        "noun_overlap":       round(noun_overlap, 6),
        "verb_overlap":       round(verb_overlap, 6),
        "adj_overlap":        round(adj_overlap, 6),
        "propn_overlap":      round(propn_overlap, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural NLP features extracted from captions"
    )
    parser.add_argument("--input",         required=True, help="Input CSV path")
    parser.add_argument("--output",        required=True, help="Output CSV path")
    parser.add_argument("--model",         default="en_core_web_trf",
                        help="spaCy model for captions (default: en_core_web_trf)")
    parser.add_argument("--article-model", default="en_core_web_sm",
                        help="spaCy model for article POS overlap (default: en_core_web_sm)")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--chunk-size",    type=int, default=500,
                        help="Rows processed per chunk to limit peak memory (default: 500)")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Process only first N rows")
    args = parser.parse_args()

    try:
        import spacy
    except ImportError:
        print("pip install spacy spacy-transformers", file=sys.stderr)
        sys.exit(1)

    try:
        nlp_cap = spacy.load(args.model)
    except OSError:
        print(f"Run: python -m spacy download {args.model}", file=sys.stderr)
        sys.exit(1)

    # Articles only need POS lemmas for overlap — sm is sufficient and far lighter.
    try:
        nlp_art = spacy.load(args.article_model)
    except OSError:
        print(f"Warning: '{args.article_model}' not found — falling back to caption model for articles.")
        nlp_art = nlp_cap

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    all_rows: list[dict] = []
    chunk_size = args.chunk_size
    n_chunks   = (len(df) + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]

        print(f"Chunk {chunk_idx + 1}/{n_chunks}  (rows {start}–{end - 1})")

        captions = chunk["caption"].fillna("").astype(str).tolist()
        articles = chunk["article_lead"].fillna("").astype(str).tolist()

        cap_docs = list(tqdm(
            nlp_cap.pipe(captions, batch_size=args.batch_size),
            total=len(captions), desc="  captions", leave=False,
        ))
        art_docs = list(tqdm(
            nlp_art.pipe(articles, batch_size=args.batch_size),
            total=len(articles), desc="  articles", leave=False,
        ))

        all_rows.extend(extract_features(c, a) for c, a in zip(cap_docs, art_docs))

        # Release docs to free memory before the next chunk.
        del cap_docs, art_docs

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(all_rows)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
