#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import json
import os

def parse_argument(val):
    try:
        return json.loads(val)
    except Exception:
        if "," in val:
            return val.split(",")
        return val

def load(filepath):
    # Load all columns (don't force index yet)
    return pd.read_csv(filepath, sep='\t')

def save(df, filepath):
    # Make sure the parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # First column becomes the index again before saving
    df.set_index(df.columns[0], inplace=True)
    df.to_csv(filepath, sep='\t', index=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Replace values in a DataFrame using pandas replace()')
    parser.add_argument('input', help='Input .tsv file (tab-separated)')
    parser.add_argument('--to_replace', required=True,
                        help='Value(s) to replace. Accepts scalar, list, or dict (as JSON string)')
    parser.add_argument('--value',
                        help='Replacement value(s). Accepts scalar, list, or dict (as JSON string). '
                             'Omit this if using nested dicts in --to_replace')
    parser.add_argument('--regex', action='store_true', help='Interpret to_replace as regex')
    parser.add_argument('--inplace', action='store_true', help='Modify the input file directly')
    parser.add_argument('-o', '--output', help='Output file path (if not using --inplace)')
    return parser.parse_args()

def main():
    args = parse_args()
    df = load(args.input)

    to_replace = parse_argument(args.to_replace)
    value = parse_argument(args.value) if args.value is not None else None

    if isinstance(to_replace, dict) and value is None:
        df_replaced = df.astype(str).replace(to_replace=to_replace, regex=args.regex)
    else:
        df_replaced = df.astype(str).replace(to_replace=to_replace, value=value, regex=args.regex)

    if args.inplace:
        save(df_replaced, args.input)
    else:
        output_path = args.output or f"{os.path.splitext(args.input)[0]}_replaced.tsv"
        save(df_replaced, output_path)

    print(df_replaced)

if __name__ == '__main__':
    main()

