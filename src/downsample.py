"""
downsample.py
-------------
Produce a smaller, equally-distributed sample from an existing pipeline CSV
(e.g. sample_90k.csv) without re-running the full preprocessing step.

Stratification is done on `category` (6 levels) and optionally also on
`source` so that each source is represented proportionally within each
category cell.

Usage:
    # 15k total — 2500 per category (default)
    python src/downsample.py --input data/samples/sample_90k.csv --output data/samples/sample_15k.csv

    # 30k total — 5000 per category
    python src/downsample.py --input data/samples/sample_90k.csv --output data/samples/sample_30k.csv --n 5000

    # 10k total — 1667 per category (rounds to even)
    python src/downsample.py --input data/samples/sample_90k.csv --output data/samples/sample_10k.csv --n 1667
"""

from __future__ import annotations

import argparse

import pandas as pd


def downsample(df: pd.DataFrame, n_per_category: int, seed: int) -> pd.DataFrame:
    """
    Sample up to n_per_category rows from each category, stratified by source
    within each category (proportional allocation).

    Returns a shuffled DataFrame.
    """
    parts: list[pd.DataFrame] = []

    for category, cat_df in df.groupby("category"):
        # Proportional allocation across sources within this category
        source_counts = cat_df["source"].value_counts()
        total_in_cat  = len(cat_df)
        allocated     = 0
        cat_parts: list[pd.DataFrame] = []

        for source, src_count in source_counts.items():
            src_df = cat_df[cat_df["source"] == source]
            # Proportional share, at least 1 if source is present
            quota = max(1, round(n_per_category * src_count / total_in_cat))
            quota = min(quota, len(src_df))  # can't exceed what's available
            cat_parts.append(src_df.sample(n=quota, random_state=seed))
            allocated += quota

        # If rounding left us short, top-up from the remaining rows
        cat_sample = pd.concat(cat_parts)
        remaining  = df[
            (df["category"] == category) & (~df.index.isin(cat_sample.index))
        ]
        shortfall = n_per_category - len(cat_sample)
        if shortfall > 0 and len(remaining) > 0:
            cat_sample = pd.concat([
                cat_sample,
                remaining.sample(n=min(shortfall, len(remaining)), random_state=seed),
            ])

        parts.append(cat_sample.head(n_per_category))

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified downsample from an existing pipeline CSV"
    )
    parser.add_argument("--input",  required=True,
                        help="Source CSV (e.g. data/samples/sample_90k.csv)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--n",      type=int, default=2500,
                        help="Rows per category (default: 2500 → 15k total for 6 categories)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    if "category" not in df.columns:
        raise ValueError("Input CSV must have a 'category' column for stratification.")

    n_cats = df["category"].nunique()
    print(f"\nCategories ({n_cats}): {sorted(df['category'].unique())}")
    print(f"Sources: {sorted(df['source'].unique())}" if "source" in df.columns else "")
    print(f"\nSampling {args.n:,} rows per category → ~{args.n * n_cats:,} total\n")

    sampled = downsample(df, args.n, args.seed)

    print("=== Output distribution ===")
    print(sampled["category"].value_counts().sort_index().to_string())
    if "source" in sampled.columns:
        print("\nBy source:")
        print(sampled.groupby(["category", "source"]).size().to_string())

    sampled.to_csv(args.output, index=False)
    print(f"\nWrote {len(sampled):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
