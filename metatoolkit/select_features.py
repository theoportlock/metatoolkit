#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
from pathlib import Path
import fnmatch

def load_table(path):
    """Load TSV file with first column as index."""
    return pd.read_csv(path, sep='\t', index_col=0)

def save_table(df, path, drop_index=False):
        """Save DataFrame as TSV, creating directories if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, sep='\t', index=not drop_index)

def parse_list(value):
    """Parse comma/newline-separated values with wildcard support"""
    if not value:
        return None
    parts = [x.strip() for x in value.replace("\r", "").split("\n")]
    items = []
    for p in parts:
        items.extend([x.strip() for x in p.split(",")])
    return [x for x in items if x]

def expand_wildcards(patterns, available):
    """Expand wildcard patterns using fnmatch over available columns/index"""
    matched = set()
    for p in patterns:
        if "*" in p or "?" in p:
            matched.update(fnmatch.filter(available, p))
        else:
            if p in available:
                matched.add(p)
    return list(matched)

def main():
    args = parse_args()

    # Load file
    df = load_table(args.input)

    # Row selections
    if args.rows:
        raw_rows = parse_list(args.rows)
        rows = expand_wildcards(raw_rows, df.index)
    else:
        rows = df.index

    # Column selections
    if args.cols:
        raw_cols = parse_list(args.cols)
        cols = expand_wildcards(raw_cols, df.columns)
    else:
        cols = df.columns

    # Apply .loc[] safely
    df_sel = df.loc[df.index.intersection(rows), df.columns.intersection(cols)]

    # Output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(args.input).stem
        output_path = f"results/{stem}_select.tsv"

    # Save
    save_table(df_sel, output_path, drop_index=args.drop_index)
    print(f"Saved {df_sel.shape[0]} rows × {df_sel.shape[1]} columns to {output_path} "
          f"(index {'dropped' if args.drop_index else 'kept'})")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select specific rows and/or columns from a TSV file using pandas .loc[] with wildcard and multiline support"
    )
    parser.add_argument("input", help="Input TSV file (with header and index column)")
    parser.add_argument("-r", "--rows", help="Comma OR newline separated list of row labels (supports wildcards *, ?)")
    parser.add_argument("-c", "--cols", help="Comma OR newline separated list of column labels (supports wildcards *, ?)")
    parser.add_argument("-o", "--output", help="Output TSV file (default: <input>_select.tsv)")
    parser.add_argument("--drop-index", action="store_true", help="Exclude index column when saving")
    return parser.parse_args()

if __name__ == "__main__":
    main()

