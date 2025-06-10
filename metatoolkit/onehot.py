#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import os
import re

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, output_path, index=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def sanitize_headers(columns):
    # Replace anything not a-z, A-Z, 0-9, - or _ with underscore
    sanitized = [re.sub(r'[^0-9a-zA-Z_\-]', '_', col) for col in columns]
    return sanitized

def onehot(df, include_cols=None, exclude_cols=None, prefix_sep='.'):
    if include_cols:
        df_to_encode = df[include_cols]
        df_remaining = df.drop(columns=include_cols)
    elif exclude_cols:
        df_to_encode = df.drop(columns=exclude_cols)
        df_remaining = df[exclude_cols]
    else:
        df_to_encode = df
        df_remaining = pd.DataFrame(index=df.index)

    df_encoded = pd.get_dummies(df_to_encode, prefix_sep=prefix_sep, dtype=bool)
    return pd.concat([df_encoded, df_remaining], axis=1)

def parse_args():
    parser = argparse.ArgumentParser(description='One-hot encode columns of a dataset.')
    parser.add_argument('subject', type=str, help='Input data file or subject name')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    parser.add_argument('--include-cols', type=str, help='Comma-separated list of columns to include in one-hot encoding')
    parser.add_argument('--exclude-cols', type=str, help='Comma-separated list of columns to exclude from one-hot encoding')
    parser.add_argument('--prefix-sep', type=str, default='.', help='Prefix separator used in one-hot encoding (default: ".")')
    parser.add_argument('--sanitize-headers', action='store_true', help='Sanitize column headers in output to letters, digits, and underscores only')
    return parser.parse_args()

def main():
    args = parse_args()

    df = load(args.subject)

    include_cols = args.include_cols.split(',') if args.include_cols else None
    exclude_cols = args.exclude_cols.split(',') if args.exclude_cols else None

    df_encoded = onehot(df, include_cols=include_cols, exclude_cols=exclude_cols, prefix_sep=args.prefix_sep)

    if args.sanitize_headers:
        df_encoded.columns = sanitize_headers(df_encoded.columns)

    print(df_encoded)

    save(df_encoded, args.output)

if __name__ == '__main__':
    main()

