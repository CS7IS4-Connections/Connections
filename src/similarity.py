"""
similarity.py
-------------
Compute three similarity measures for each caption–article pair.

Preprocessing (shared for TF-IDF and Jaccard):
  spaCy lemmatisation, lowercase, stopword and punctuation removal.

  tfidf_sim  : TF-IDF cosine on lemmatised text
               (sklearn, bigrams, min_df=2, max_df=0.95,
                vectoriser fit on combined caption+article corpus)
  jaccard_sim: |C ∩ A| / |C ∪ A| on lemma sets; 0.0 if union is empty.
  sbert_sim  : Cosine similarity of all-MiniLM-L6-v2 embeddings.
               Articles truncated to 128 tokens before encoding;
               TF-IDF and Jaccard use the full 512-token article_lead.

Usage:
    python src/similarity.py \\
        --input  results/structural.csv \\
        --output results/similarity.csv

    # If your CSV uses different column names:
    python src/similarity.py \\
        --input results/structural.csv --output results/similarity.csv \\
        --caption-col caption --article-col article_lead
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared spaCy preprocessing
# ---------------------------------------------------------------------------

def _lemmatised_text(doc) -> str:
    """Lowercase lemmas, stopwords and punctuation removed — for TF-IDF."""
    return " ".join(
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha
    )


def _lemma_set(doc) -> set[str]:
    """Lemma set for Jaccard — same filter as _lemmatised_text."""
    return {
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha
    }


def preprocess(texts: list[str], nlp, batch_size: int, desc: str):
    """Run spaCy over texts; return (lemma_texts, lemma_sets)."""
    lemma_texts, lemma_sets = [], []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts),
                    desc=desc, leave=False):
        lemma_texts.append(_lemmatised_text(doc))
        lemma_sets.append(_lemma_set(doc))
    return lemma_texts, lemma_sets


# ---------------------------------------------------------------------------
# Similarity methods
# ---------------------------------------------------------------------------

def tfidf_cosine(
    cap_lemma: list[str], art_lemma: list[str]
) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    vect.fit(cap_lemma + art_lemma)
    cap_vecs = vect.transform(cap_lemma)
    art_vecs = vect.transform(art_lemma)

    sims = np.zeros(len(cap_lemma))
    for i in tqdm(range(len(cap_lemma)), desc="TF-IDF cosine", leave=False):
        sims[i] = sk_cos(cap_vecs[i], art_vecs[i])[0, 0]
    return sims


def jaccard_sim(
    cap_sets: list[set[str]], art_sets: list[set[str]]
) -> np.ndarray:
    sims = np.zeros(len(cap_sets))
    for i, (cs, as_) in enumerate(zip(cap_sets, art_sets)):
        union = cs | as_
        sims[i] = len(cs & as_) / len(union) if union else 0.0
    return sims


def sbert_cosine(
    captions: list[str], articles: list[str], batch_size: int,
    sbert_article_tokens: int = 128,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    # Truncate articles to sbert_article_tokens whitespace tokens
    articles_trunc = [" ".join(a.split()[:sbert_article_tokens]) for a in articles]

    model    = SentenceTransformer("all-MiniLM-L6-v2")
    cap_embs = model.encode(
        captions, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    art_embs = model.encode(
        articles_trunc, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.sum(cap_embs * art_embs, axis=1).astype(float)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, cat_col: str | None) -> None:
    sim_cols = ["tfidf_sim", "jaccard_sim", "sbert_sim"]
    print("\n=== Similarity Score Summary ===")
    for col in sim_cols:
        s = df[col].describe()
        print(f"  {col}: mean={s['mean']:.4f}  std={s['std']:.4f}"
              f"  min={s['min']:.4f}  max={s['max']:.4f}")

    if "source" in df.columns:
        print("\nMean by source:")
        print(df.groupby("source")[sim_cols].mean().round(4).to_string())

    if cat_col and cat_col in df.columns:
        print(f"\nMean by {cat_col}:")
        print(df.groupby(cat_col)[sim_cols].mean().round(4).to_string())

    # Flag potential outliers (sbert_sim < 0.05 — effectively unrelated pairs)
    low = (df["sbert_sim"] < 0.05).sum()
    if low:
        print(f"\n  Notable: {low} rows have sbert_sim < 0.05 "
              f"({100 * low / len(df):.1f}% of pairs — likely caption/article mismatch)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Caption–article similarity scores")
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--model",           default="en_core_web_trf")
    parser.add_argument("--batch-size",      type=int, default=64)
    parser.add_argument("--limit",           type=int, default=None)
    parser.add_argument("--caption-col",     default="caption",
                        help="CSV column containing caption text")
    parser.add_argument("--article-col",     default="article_lead",
                        help="CSV column containing article text")
    parser.add_argument("--sbert-art-tokens", type=int, default=128,
                        help="Max article tokens passed to SBERT (default 128)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    for col in (args.caption_col, args.article_col):
        if col not in df.columns:
            print(f"ERROR: column '{col}' not found. "
                  f"Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    captions = df[args.caption_col].fillna("").astype(str).tolist()
    articles = df[args.article_col].fillna("").astype(str).tolist()

    # --- Load spaCy once ---
    try:
        import spacy
        nlp = spacy.load(args.model)
    except Exception as e:
        print(f"spaCy load error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Shared preprocessing (TF-IDF + Jaccard) ---
    print("Preprocessing captions ...")
    cap_lemma, cap_sets = preprocess(captions, nlp, args.batch_size, "captions")
    print("Preprocessing articles ...")
    art_lemma, art_sets = preprocess(articles, nlp, args.batch_size, "articles")

    # --- Similarity scores ---
    print("Computing TF-IDF cosine similarity ...")
    df["tfidf_sim"] = tfidf_cosine(cap_lemma, art_lemma)

    print("Computing Jaccard similarity ...")
    df["jaccard_sim"] = jaccard_sim(cap_sets, art_sets)

    print("Computing SBERT similarity (articles truncated to "
          f"{args.sbert_art_tokens} tokens) ...")
    df["sbert_sim"] = sbert_cosine(captions, articles, args.batch_size,
                                   args.sbert_art_tokens)

    # --- Summary ---
    cat_col = next((c for c in ("category", "topic") if c in df.columns), None)
    print_summary(df, cat_col)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
