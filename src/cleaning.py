"""
cleaning.py
-----------
Remove rows with empty article_lead from a preprocessing CSV.

Usage:
    python src/cleaning.py --input data/samples/sample_5k.csv \
        --output data/samples/sample_5k_clean.csv
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Drop rows where article_lead is empty or missing."
    )
    parser.add_argument("--input",  required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    before = len(df)

    df = df[df["article_lead"].notna() & (df["article_lead"].str.strip() != "")]

    after = len(df)
    dropped = before - after

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Rows before : {before:,}")
    print(f"Rows dropped: {dropped:,} ({100 * dropped / before:.1f}%)")
    print(f"Rows after  : {after:,}")
    print(f"Output      : {args.output}")


if __name__ == "__main__":
    main()
