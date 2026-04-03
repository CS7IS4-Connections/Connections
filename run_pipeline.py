"""
run_pipeline.py
---------------
Orchestrate the full text-analytics pipeline in order:

  1. src/structural_features.py  — caption structural features + POS overlap
  2. src/similarity.py           — TF-IDF cosine, Jaccard, SBERT similarity
  3. src/entity_alignment.py     — named-entity overlap
  4. src/caption_classifier.py   — caption type label (Marsh & White 2003)

Each step reads the CSV produced by the previous step.  Intermediate files
are written alongside the final output so each step can be inspected or
re-run independently.

If the input CSV contains an `article_text` column but no `article_lead`
column, it is renamed to `article_lead` before being passed downstream.

Usage:
    python run_pipeline.py --input data/samples/sample_60k.csv --output results/sample_results.csv
    python run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_count(path: str) -> int:
    """Fast line count (excludes header)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh) - 1


def run_step(step_n: int, script: str, input_path: str, output_path: str,
             extra_args: list[str] | None = None) -> None:
    n_rows = _row_count(input_path)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Step {step_n}/4: {script} — processing {n_rows:,} rows…")
    print(sep)
    cmd = [sys.executable, script, "--input", input_path, "--output", output_path]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full text-analytics pipeline")
    parser.add_argument("--input",    default="data/samples/sample_60k.csv",
                        help="Input CSV (expects caption + article_lead or article_text columns)")
    parser.add_argument("--output",   default="results/sample_results.csv")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Process only the first 500 rows (smoke test)")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    # ── Column normalisation ────────────────────────────────────────────────
    # If the input file has article_text instead of article_lead, rename it
    # so that all downstream scripts see the expected column name.
    df_in = pd.read_csv(args.input)
    if "article_lead" not in df_in.columns and "article_text" in df_in.columns:
        print("Note: renaming column 'article_text' → 'article_lead' for downstream compatibility.")
        df_in = df_in.rename(columns={"article_text": "article_lead"})

    if args.dry_run:
        print("DRY-RUN mode: limiting to first 500 rows.")
        df_in = df_in.head(500).copy()

    # Write the (possibly modified) input to a staging path
    base         = os.path.splitext(os.path.abspath(args.output))[0]
    staged_input = base + "_step0_input.csv"
    df_in.to_csv(staged_input, index=False)

    # Intermediate paths
    s1 = base + "_step1_structural.csv"
    s2 = base + "_step2_similarity.csv"
    s3 = base + "_step3_entity.csv"

    # ── Run stages ──────────────────────────────────────────────────────────
    run_step(1, "src/structural_features.py", staged_input, s1)
    run_step(2, "src/similarity.py",          s1,           s2)
    run_step(3, "src/entity_alignment.py",    s2,           s3)
    run_step(4, "src/caption_classifier.py",  s3,           args.output)

    # ── Final summary ───────────────────────────────────────────────────────
    df = pd.read_csv(args.output)
    sep = "=" * 60
    print(f"\n{sep}\nPIPELINE COMPLETE\n{sep}")
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Null counts (only columns with at least one null)
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("Nulls  : none")
    else:
        print("Nulls  :")
        for col, n in nulls.items():
            print(f"  {col:<30s} {n:>6,}")

    # Similarity stats
    for col in ("tfidf_sim", "jaccard_sim", "sbert_sim"):
        if col in df.columns:
            print(f"  {col:<20s}  mean={df[col].mean():.4f}"
                  f"  min={df[col].min():.4f}  max={df[col].max():.4f}")

    # Categorical distributions
    for col in ("source", "category", "caption_type"):
        if col in df.columns:
            print(f"\n{col} distribution:")
            vc = df[col].value_counts()
            for val, n in vc.items():
                print(f"  {str(val):<30s} {n:>6,}  ({100 * n / len(df):.1f}%)")

    print(f"\nOutput : {args.output}")


if __name__ == "__main__":
    main()
