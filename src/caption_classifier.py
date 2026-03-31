"""
caption_classifier.py
---------------------
PLACEHOLDER — classification logic is not yet implemented.

Adds a caption_type column with the value "unclassified" for every row.

TODO: Implement classify() once the caption category definitions are finalised.
      The function signature and column name should remain stable so that
      downstream pipeline steps do not need to change.

Usage:
    python src/caption_classifier.py \\
        --input  results/entity.csv \\
        --output results/sample_results.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def classify(caption: str) -> str:
    # TODO: implement classification logic once category definitions are finalised.
    return "unclassified"


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption classifier (placeholder)")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    df["caption_type"] = df["caption"].fillna("").apply(classify)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
