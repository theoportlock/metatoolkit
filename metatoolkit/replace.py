#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import json
import os

def parse_argument(val):
    # Attempt to parse value as JSON-like dict/list
    try:
        return json.loads(val)
    except Exception:
        # Fallback: treat as string or comma-separated list
        if "," in val:
            return val.split(",")
        return val

def load(filepath):
    return pd.read_csv(filepath, sep='\t', index_col=0)

def save(df, filepath, index=True):
    df.to_csv(filepath, sep='\t', index=index)

def parse_args():
    parser = argparse.ArgumentParser(description='Replace values in a DataFrame using pandas replace()')
    parser.add_argument('input', help='Input .tsv file (tab-separated, with index column)')
    parser.add_argument('--to_replace', required=True, help='Value(s) to replace. Accepts scalar, list, or dict (as JSON string)')
    parser.add_argument('--value', default=None, help='Replacement value(s). Can be scalar, list, or dict (as JSON string)')
    parser.add_argument('--regex', action='store_true', help='Interpret to_replace as regex')
    parser.add_argument('--inplace', action='store_true', help='Modify the input file directly')
    parser.add_argument('--output', help='Output file path (if not using --inplace)')
    return parser.parse_args()

def main():
    args = parse_args()
    df = load(args.input)

    # Parse arguments
    to_replace = parse_argument(args.to_replace)
    value = parse_argument(args.value) if args.value is not None else None

    df_replaced = df.replace(
        to_replace=to_replace,
        value=value,
        regex=args.regex
    )

    if args.inplace:
        save(df_replaced, args.input)
    else:
        output_path = args.output or f"{os.path.splitext(args.input)[0]}_replaced.tsv"
        save(df_replaced, output_path)

    print(df_replaced)

if __name__ == '__main__':
    main()

