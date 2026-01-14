#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fill missing values in a TSV file with a specified value.

Examples:
    # Fill all missing cells with 0
    python fillna.py -i data.tsv -v 0 -o results/data_filled.tsv

    # Fill missing values only in specific columns
    python fillna.py -i data.tsv -c Age,Height,Weight -v 0 -o results/data_filled.tsv

    # Fill with a string value
    python fillna.py -i Demographics.tsv -v "Unknown" -o results/Demographics_filled.tsv
"""

import argparse
import pandas as pd
import os
from pathlib import Path


def load_table(path):
    """Load TSV file with first column as index."""
    return pd.read_csv(path, sep="\t", index_col=0)


def save_table(df, path):
    """Save DataFrame as TSV, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t")


def fill_missing(df, value, columns=None):
    """Fill missing values either for specific columns or all columns."""
    if columns:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in input file: {', '.join(missing_cols)}")
        df[columns] = df[columns].fillna(value)
    else:
        df = df.fillna(value)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing values in a TSV file with a specified value."
    )
    parser.add_argument("-i", "--input", required=True, help="Input TSV file")
    parser.add_argument(
        "-c", "--column",
        help="Optional: comma-separated list of columns to fill (default: all columns)"
    )
    parser.add_argument("-v", "--value", required=True, help="Value to fill missing cells with")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse column argument into list (if provided)
    columns = [c.strip() for c in args.column.split(",")] if args.column else None

    # Load
    df = load_table(args.input)

    # Fill missing values
    df_filled = fill_missing(df, args.value, columns)

    # Save to specified output path
    save_table(df_filled, args.output)

    if columns:
        print(f"✅ Filled missing values in columns {', '.join(columns)} with '{args.value}' → {args.output}")
    else:
        print(f"✅ Filled all missing values with '{args.value}' → {args.output}")


if __name__ == "__main__":
    main()

