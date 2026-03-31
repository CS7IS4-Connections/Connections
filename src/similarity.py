"""
similarity.py
-------------
Compute three similarity measures for each caption–article_lead pair.

  tfidf_sim   : TF-IDF cosine (sklearn, bigrams, min_df=2, max_df=0.95)
                Vectoriser is fit on the combined caption+article corpus.
  jaccard_sim : Lemma-set Jaccard; stopwords and punctuation removed (spaCy).
  sbert_sim   : Semantic cosine similarity via sentence-transformers
                (all-MiniLM-L6-v2, batch_size=64).

Usage:
    pip install scikit-learn sentence-transformers
    python -m spacy download en_core_web_trf
    python src/similarity.py \\
        --input  results/structural.csv \\
        --output results/similarity.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def tfidf_cosine(captions: list[str], articles: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    vect.fit(captions + articles)
    cap_vecs = vect.transform(captions)
    art_vecs = vect.transform(articles)

    sims = np.zeros(len(captions))
    for i in tqdm(range(len(captions)), desc="TF-IDF cosine", leave=False):
        sims[i] = sk_cos(cap_vecs[i], art_vecs[i])[0, 0]
    return sims


def jaccard_lemma(
    captions: list[str], articles: list[str], nlp, batch_size: int
) -> np.ndarray:
    n    = len(captions)
    docs = list(tqdm(
        nlp.pipe(captions + articles, batch_size=batch_size),
        total=2 * n,
        desc="spaCy lemmatise",
        leave=False,
    ))
    cap_docs = docs[:n]
    art_docs = docs[n:]

    def lemma_set(doc) -> set[str]:
        return {
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha
        }

    sims = np.zeros(n)
    for i, (cd, ad) in enumerate(zip(cap_docs, art_docs)):
        cs    = lemma_set(cd)
        as_   = lemma_set(ad)
        union = cs | as_
        sims[i] = len(cs & as_) / len(union) if union else 0.0
    return sims


def sbert_cosine(captions: list[str], articles: list[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model    = SentenceTransformer("all-MiniLM-L6-v2")
    cap_embs = model.encode(
        captions, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    art_embs = model.encode(
        articles, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    # normalised embeddings → cosine similarity == dot product
    return np.sum(cap_embs * art_embs, axis=1).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption–article similarity scores")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model",      default="en_core_web_trf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    captions = df["caption"].fillna("").astype(str).tolist()
    articles = df["article_lead"].fillna("").astype(str).tolist()

    print("Computing TF-IDF cosine similarity ...")
    df["tfidf_sim"] = tfidf_cosine(captions, articles)

    print("Computing Jaccard similarity ...")
    try:
        import spacy
        nlp = spacy.load(args.model)
    except Exception as e:
        print(f"spaCy load error: {e}", file=sys.stderr)
        sys.exit(1)
    df["jaccard_sim"] = jaccard_lemma(captions, articles, nlp, args.batch_size)

    print("Computing SBERT similarity ...")
    df["sbert_sim"] = sbert_cosine(captions, articles, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
