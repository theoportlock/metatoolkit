#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
from pathlib import Path

def load_table(path):
    """Load TSV file with first column as index."""
    return pd.read_csv(path, sep='\t', index_col=0)

def save_table(df, path):
    """Save DataFrame as TSV, creating directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select specific rows and/or columns from a TSV file using pandas .loc[]"
    )
    parser.add_argument("input", help="Input TSV file (with header and index column)")
    parser.add_argument("-r", "--rows", help="Comma-separated list of row labels to keep")
    parser.add_argument("-c", "--cols", help="Comma-separated list of column labels to keep")
    parser.add_argument("-o", "--output", help="Output TSV file (default: <input>_select.tsv)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load file
    df = load_table(args.input)

    # Row and column selections
    rows = args.rows.split(",") if args.rows else df.index
    cols = args.cols.split(",") if args.cols else df.columns

    # Apply .loc[] safely
    df_sel = df.loc[df.index.intersection(rows), df.columns.intersection(cols)]

    # Output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.input).stem
        output_path = f"results/{stem}_select.tsv"

    # Save
    save_table(df_sel, output_path)
    print(f"Saved {df_sel.shape[0]} rows × {df_sel.shape[1]} columns to {output_path}")

if __name__ == "__main__":
    main()

