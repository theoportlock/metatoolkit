#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fill missing values in a TSV file with a specified value.

Examples:
    # Fill all missing cells with 0
    python fillna.py -i data.tsv -v 0 -o results/data_filled.tsv

    # Fill missing values only in one column
    python fillna.py -i data.tsv -c Age -v 0 -o results/data_filled.tsv

    # Fill with string value
    python fillna.py -i Demographics.tsv -v "Unknown"
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


def fill_missing(df, value, column=None):
    """Fill missing values either for a specific column or all columns."""
    if column:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in input file.")
        df[column] = df[column].fillna(value)
    else:
        df = df.fillna(value)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing values in a TSV file with a specified value."
    )
    parser.add_argument("-i", "--input", required=True, help="Input TSV file")
    parser.add_argument(
        "-c", "--column", help="Optional: specific column name to fill (default: all columns)"
    )
    parser.add_argument("-v", "--value", required=True, help="Value to fill missing cells with")
    parser.add_argument(
        "-o", "--output", help="Output TSV file (default: <input>_filled.tsv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load
    df = load_table(args.input)

    # Fill missing values
    df_filled = fill_missing(df, args.value, args.column)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.input).stem
        output_path = f"results/{stem}_filled.tsv"

    # Save
    save_table(df_filled, output_path)

    if args.column:
        print(f"✅ Filled missing values in column '{args.column}' with '{args.value}' → {output_path}")
    else:
        print(f"✅ Filled all missing values with '{args.value}' → {output_path}")


if __name__ == "__main__":
    main()

