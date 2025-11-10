#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import glob
import sys

def load(path):
    """Load a TSV file (must exist)."""
    path = path.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"⚠️ File not found: {path}")
    return pd.read_csv(path, sep='\t', index_col=0)

def save(df, path, index=True):
    """Save a DataFrame as TSV, ensuring the directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=index)
    print(f"💾 Saved {df.shape[0]} rows × {df.shape[1]} columns to {path}")

def merge(datasets, join='inner', append=False, add_filename_column=False,
          filenames=None, filename_format='base',
          add_prefix=False, sep=':'):
    """Merge (columns) or append (rows) multiple TSV files."""
    if join not in ('inner', 'outer'):
        raise ValueError("join must be 'inner' or 'outer'")
    if filename_format not in ('path', 'base'):
        raise ValueError("filename_format must be 'path' or 'base'")

    # Optionally prefix columns with dataset basename
    if add_prefix and filenames is not None and not append:
        prefixed_datasets = []
        for df, fname in zip(datasets, filenames):
            dataset_name = os.path.basename(fname) if filename_format == 'base' else fname
            dataset_name = os.path.splitext(dataset_name)[0]
            df.columns = [f"{dataset_name}{sep}{col}" for col in df.columns]
            prefixed_datasets.append(df)
        datasets = prefixed_datasets

    if append:
        if add_filename_column and filenames is not None:
            for df, fname in zip(datasets, filenames):
                label = os.path.basename(fname) if filename_format == 'base' else fname
                df['filename'] = os.path.splitext(label)[0]
        return pd.concat(datasets, axis=0, join=join)
    else:
        return pd.concat(datasets, axis=1, join=join)

def expand_wildcards(paths):
    """Expand wildcard expressions (e.g., *.tsv) into actual file lists."""
    expanded = []
    for p in paths:
        matches = glob.glob(p)
        if not matches:
            print(f"⚠️ No files matched: {p}", file=sys.stderr)
        expanded.extend(sorted(matches))
    return expanded

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge multiple TSV datasets horizontally or vertically.'
    )
    parser.add_argument('datasets', nargs='+', help='List of dataset TSV files (supports wildcards)')
    parser.add_argument('-j', '--join', default='inner', choices=['inner', 'outer'],
                        help='Join method: inner (default) or outer')
    parser.add_argument('-a', '--append', action='store_true',
                        help='Append vertically instead of joining columns')
    parser.add_argument('--add-filename', action='store_true',
                        help='Add a filename column when appending')
    parser.add_argument('--filename-format', choices=['path', 'base'], default='base',
                        help='Format for filename column: base (default) or path')
    parser.add_argument('--add-prefix', action='store_true',
                        help='Add dataset basename as prefix to column names (only when merging columns)')
    parser.add_argument('--sep', default=':',
                        help='Separator between dataset basename and column name (default: ":")')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file path (required)')
    return parser.parse_args()

def main():
    args = parse_args()
    expanded = expand_wildcards(args.datasets)
    if not expanded:
        sys.exit("❌ No valid input files found.")
    dfs = [load(path) for path in expanded]
    result = merge(
        datasets=dfs,
        join=args.join,
        append=args.append,
        add_filename_column=args.add_filename,
        filenames=expanded,
        filename_format=args.filename_format,
        add_prefix=args.add_prefix,
        sep=args.sep
    )
    save(result, args.output)

if __name__ == '__main__':
    main()

