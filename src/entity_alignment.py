"""
entity_alignment.py
-------------------
Named-entity alignment between captions and article_lead text.

Uses spaCy en_core_web_trf for NER and rapidfuzz for Levenshtein fuzzy
matching (threshold 0.85).  Entity types: PERSON, GPE, ORG, DATE.

Metrics per row:
  person_overlap  — |E_cap ∩ E_art| / |E_cap|  for PERSON  (0.0 if empty)
  gpe_overlap     — same for GPE
  org_overlap     — same for ORG
  date_overlap    — same for DATE
  entity_jaccard  — overall |E_cap ∩ E_art| / |E_cap ∪ E_art|
  entity_coverage — overall |E_cap ∩ E_art| / |E_art|

Checkpointing: the script saves a partial output after every chunk so that
a timed-out run can be resumed without losing progress.

Usage:
    # Fresh run
    python src/entity_alignment.py --input results/similarity.csv --output results/entity.csv

    # Resume after a timeout (reads .partial checkpoint automatically)
    python src/entity_alignment.py --input results/similarity.csv --output results/entity.csv --resume
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

ENTITY_TYPES    = ("PERSON", "GPE", "ORG", "DATE")
FUZZY_THRESHOLD = 0.85


def _lev_sim(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / max_len


def _has_match(entity: str, candidates: list[str]) -> bool:
    e = entity.lower()
    return any(_lev_sim(e, c.lower()) >= FUZZY_THRESHOLD for c in candidates)


def _ents_by_type(doc) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {t: [] for t in ENTITY_TYPES}
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            result[ent.label_].append(ent.text.strip())
    return result


def _compute_metrics(cap_ents: dict[str, list[str]], art_ents: dict[str, list[str]]) -> dict:
    row: dict = {}
    n_matched_total = 0
    n_cap_total     = 0
    n_art_total     = 0

    for etype in ENTITY_TYPES:
        ce = cap_ents[etype]
        ae = art_ents[etype]
        n_matched = sum(1 for e in ce if _has_match(e, ae))

        row[f"{etype.lower()}_overlap"] = n_matched / len(ce) if ce else 0.0

        n_matched_total += n_matched
        n_cap_total     += len(ce)
        n_art_total     += len(ae)

    union_size = n_cap_total + n_art_total - n_matched_total
    row["entity_jaccard"]  = n_matched_total / union_size  if union_size > 0  else 0.0
    row["entity_coverage"] = n_matched_total / n_art_total if n_art_total > 0 else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Named-entity alignment for caption–article pairs"
    )
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--model",      default="en_core_web_trf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Rows per NER chunk to limit peak memory (default: 500)")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from checkpoint if one exists for this output path")
    args = parser.parse_args()

    try:
        from rapidfuzz.distance import Levenshtein  # noqa: F401
    except ImportError:
        print("pip install rapidfuzz", file=sys.stderr)
        sys.exit(1)

    try:
        import spacy
        nlp = spacy.load(args.model)
    except Exception as e:
        print(f"spaCy load error: {e}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    # ── Checkpoint ─────────────────────────────────────────────────────────
    checkpoint_path = args.output + ".partial"
    done_df: pd.DataFrame | None = None
    start_row = 0

    if args.resume and os.path.exists(checkpoint_path):
        done_df   = pd.read_csv(checkpoint_path)
        start_row = len(done_df)
        print(f"Resuming from row {start_row:,} "
              f"({start_row // args.chunk_size} chunks already done, "
              f"{len(df) - start_row:,} remaining)")
    elif args.resume:
        print("--resume passed but no checkpoint found; starting from scratch.")

    # Work only on the unprocessed tail of the dataframe
    remaining_df = df.iloc[start_row:].reset_index(drop=True)
    captions     = remaining_df["caption"].fillna("").astype(str).tolist()
    articles     = remaining_df["article_lead"].fillna("").astype(str).tolist()

    new_rows:  list[dict] = []
    chunk_size = args.chunk_size
    n_chunks   = (len(remaining_df) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc="Entity alignment chunks"):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, len(remaining_df))

        cap_docs = list(nlp.pipe(captions[start:end], batch_size=args.batch_size))
        art_docs = list(nlp.pipe(articles[start:end], batch_size=args.batch_size))

        new_rows.extend(
            _compute_metrics(_ents_by_type(cd), _ents_by_type(ad))
            for cd, ad in zip(cap_docs, art_docs)
        )
        del cap_docs, art_docs

        # ── Save checkpoint after every chunk ──────────────────────────────
        processed_so_far = pd.concat(
            [remaining_df.iloc[:len(new_rows)].reset_index(drop=True),
             pd.DataFrame(new_rows)],
            axis=1,
        )
        checkpoint_out = (
            pd.concat([done_df, processed_so_far], ignore_index=True)
            if done_df is not None
            else processed_so_far
        )
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        checkpoint_out.to_csv(checkpoint_path, index=False)

    # ── Combine done + new and write final output ───────────────────────────
    new_out = pd.concat(
        [remaining_df.reset_index(drop=True), pd.DataFrame(new_rows)],
        axis=1,
    )
    out = (
        pd.concat([done_df, new_out], ignore_index=True)
        if done_df is not None
        else new_out
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} rows -> {args.output}")

    # Clean up checkpoint once final output is written
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
