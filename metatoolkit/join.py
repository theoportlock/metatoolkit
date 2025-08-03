#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Theo Portlock
Join two TSV files using pandas merge with optional smart handling of identical suffixed columns.
"""

import argparse
import pandas as pd
import os

def load(filepath):
    """Load TSV file from path or from 'results/' subfolder if not a full path."""
    if os.path.isfile(filepath):
        return pd.read_csv(filepath, sep='\t', index_col=0)
    return pd.read_csv(f'results/{filepath}.tsv', sep='\t', index_col=0)

def save(df, name):
    """Save DataFrame"""
    os.makedirs(os.path.dirname(name), exist_ok=True)
    df.to_csv(name, sep='\t')

def combine_identical_columns(df, suffixes):
    """Drop duplicated suffixed columns if values are identical, restoring base name."""
    suffix1, suffix2 = suffixes
    to_drop, to_add = [], {}

    for base in set(col.removesuffix(suffix1).removesuffix(suffix2) for col in df.columns):
        col1, col2 = f'{base}{suffix1}', f'{base}{suffix2}'
        if col1 in df.columns and col2 in df.columns and df[col1].equals(df[col2]):
            to_add[base] = df[col1]
            to_drop.extend([col1, col2])

    return df.drop(columns=to_drop).assign(**to_add)

def join_dataframes(file1, file2, how='inner', on=None, left_on=None, right_on=None, suffixes=('_x', '_y')):
    """Join two DataFrames loaded from file1 and file2 using the specified merge options."""
    df1 = load(file1)
    df2 = load(file2)

    merge_kwargs = {'how': how, 'suffixes': suffixes}

    # Handle join keys
    if on:
        merge_kwargs['on'] = on
        df1 = df1.reset_index()
        df2 = df2.reset_index()
    elif left_on and right_on:
        merge_kwargs['left_on'] = left_on
        merge_kwargs['right_on'] = right_on
        df1 = df1.reset_index()
        df2 = df2.reset_index()
    else:
        # Join on index — do NOT reset index
        merge_kwargs['left_index'] = True
        merge_kwargs['right_index'] = True

    merged = pd.merge(df1, df2, **merge_kwargs)

    # Clean up duplicated columns if identical
    merged = combine_identical_columns(merged, suffixes)

    # If merged on columns, restore index from first column
    if 'left_index' not in merge_kwargs:
        merged.set_index(merged.columns[0], inplace=True)

    return merged


def parse_args():
    parser = argparse.ArgumentParser(description='Join two dataframes using pandas merge.')
    parser.add_argument('file1', help='First input file.')
    parser.add_argument('file2', help='Second input file.')
    parser.add_argument('-o', '--output', default='joined', help='Output filename prefix.')
    parser.add_argument('--how', default='inner', choices=['inner', 'outer', 'left', 'right', 'cross'],
                        help='Merge method.')
    parser.add_argument('--on', help='Column(s) to join on (comma-separated).')
    parser.add_argument('--left_on', help='Left join key(s) (comma-separated).')
    parser.add_argument('--right_on', help='Right join key(s) (comma-separated).')
    parser.add_argument('--suffixes', default='_x,_y', help='Suffixes for overlapping columns (comma-separated).')
    return parser.parse_args()

def main():
    args = parse_args()
    suffixes = tuple(args.suffixes.split(','))
    if len(suffixes) != 2:
        raise ValueError("Please provide exactly two comma-separated suffixes (e.g., '_x,_y')")

    merged_df = join_dataframes(
        file1=args.file1,
        file2=args.file2,
        how=args.how,
        on=args.on.split(',') if args.on else None,
        left_on=args.left_on.split(',') if args.left_on else None,
        right_on=args.right_on.split(',') if args.right_on else None,
        suffixes=suffixes
    )
    save(merged_df, args.output)

if __name__ == '__main__':
    main()

