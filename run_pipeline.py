"""
run_pipeline.py
---------------
Orchestrate the full text-analytics pipeline in order:

  1. src/structural_features.py  — caption structural features + POS overlap
  2. src/similarity.py           — TF-IDF cosine, Jaccard, SBERT similarity
  3. src/entity_alignment.py     — named-entity overlap
  4. src/caption_classifier.py   — caption type label (placeholder)

Each step reads the CSV produced by the previous step.  Intermediate files
are written alongside the final output so each step can be inspected or
re-run independently.

Usage:
    python run_pipeline.py \\
        --input  data/samples/sample_data_clean.csv \\
        --output results/sample_results.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd


def run_step(script: str, input_path: str, output_path: str) -> None:
    cmd = [sys.executable, script, "--input", input_path, "--output", output_path]
    sep = "=" * 60
    print(f"\n{sep}\nStep: {script}\n{sep}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full text-analytics pipeline")
    parser.add_argument("--input",  default="data/samples/sample_data_clean.csv")
    parser.add_argument("--output", default="results/sample_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Intermediate files sit in the same directory as the final output.
    base = os.path.splitext(os.path.abspath(args.output))[0]
    s1   = base + "_step1_structural.csv"
    s2   = base + "_step2_similarity.csv"
    s3   = base + "_step3_entity.csv"

    run_step("src/structural_features.py", args.input, s1)
    run_step("src/similarity.py",          s1,         s2)
    run_step("src/entity_alignment.py",    s2,         s3)
    run_step("src/caption_classifier.py",  s3,         args.output)

    df = pd.read_csv(args.output)
    sep = "=" * 60
    print(f"\n{sep}\nPIPELINE COMPLETE\n{sep}")
    print(f"Rows : {len(df):,}")
    print(f"Cols : {len(df.columns)}")
    for col in ("tfidf_sim", "jaccard_sim", "sbert_sim"):
        if col in df.columns:
            print(
                f"  {col:20s}  mean={df[col].mean():.4f}"
                f"  min={df[col].min():.4f}  max={df[col].max():.4f}"
            )
    if "caption_type" in df.columns:
        print("\ncaption_type distribution:")
        print(df["caption_type"].value_counts().to_string())
    print(f"\nOutput : {args.output}")


if __name__ == "__main__":
    main()
