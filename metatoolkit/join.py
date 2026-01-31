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


def _get_join_keys(df, on=None, left_on=None, right_on=None, use_index=False):
    """Return a Series of join keys as tuples (robust to multi-key joins)."""
    if use_index:
        return df.index.to_series().apply(lambda x: (x,))
    if on:
        return df[on].apply(tuple, axis=1)
    if left_on:
        return df[left_on].apply(tuple, axis=1)
    raise RuntimeError("Unable to determine join keys")


def print_join_stats(stats):
    print("Join summary")
    print("------------")
    for label, s in stats.items():
        print(
            f"{label}: "
            f"{s['lost']} / {s['total']} rows lost "
            f"({s['pct_lost']:.2f}%)"
        )


def join_dataframes(file1, file2, how='inner',
                    on=None, left_on=None, right_on=None,
                    suffixes=('_x', '_y')):

    df1 = load(file1)
    df2 = load(file2)

    n1, n2 = len(df1), len(df2)

    merge_kwargs = {'how': how, 'suffixes': suffixes}

    # Handle join keys
    if on:
        merge_kwargs['on'] = on
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        use_index = False
    elif left_on and right_on:
        merge_kwargs['left_on'] = left_on
        merge_kwargs['right_on'] = right_on
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        use_index = False
    else:
        merge_kwargs['left_index'] = True
        merge_kwargs['right_index'] = True
        use_index = True

    merged = pd.merge(df1, df2, **merge_kwargs)

    # Compute join loss statistics
    k1 = _get_join_keys(df1, on=on, left_on=left_on, use_index=use_index)
    k2 = _get_join_keys(df2, on=on, left_on=right_on, use_index=use_index)
    km = _get_join_keys(merged, on=on, left_on=left_on, use_index=use_index)

    k1_set, k2_set, km_set = set(k1), set(k2), set(km)

    stats = {
        'df1': {
            'total': n1,
            'lost': len(k1_set - km_set),
            'pct_lost': 100 * len(k1_set - km_set) / n1 if n1 else 0
        },
        'df2': {
            'total': n2,
            'lost': len(k2_set - km_set),
            'pct_lost': 100 * len(k2_set - km_set) / n2 if n2 else 0
        }
    }

    # Clean duplicated columns
    merged = combine_identical_columns(merged, suffixes)

    # Restore index if needed
    if 'left_index' not in merge_kwargs:
        merged.set_index(merged.columns[0], inplace=True)

    return merged, stats


def parse_args():
    parser = argparse.ArgumentParser(description='Join two dataframes using pandas merge.')
    parser.add_argument('file1', help='First input file.')
    parser.add_argument('file2', help='Second input file.')
    parser.add_argument('-o', '--output', default='joined', help='Output filename prefix.')
    parser.add_argument('--how', default='inner',
                        choices=['inner', 'outer', 'left', 'right', 'cross'],
                        help='Merge method.')
    parser.add_argument('--on', help='Column(s) to join on (comma-separated).')
    parser.add_argument('--left_on', help='Left join key(s) (comma-separated).')
    parser.add_argument('--right_on', help='Right join key(s) (comma-separated).')
    parser.add_argument('--suffixes', default='_x,_y',
                        help='Suffixes for overlapping columns (comma-separated).')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress join summary output.')
    return parser.parse_args()


def main():
    args = parse_args()
    suffixes = tuple(args.suffixes.split(','))
    if len(suffixes) != 2:
        raise ValueError("Please provide exactly two comma-separated suffixes (e.g., '_x,_y')")

    merged_df, stats = join_dataframes(
        file1=args.file1,
        file2=args.file2,
        how=args.how,
        on=args.on.split(',') if args.on else None,
        left_on=args.left_on.split(',') if args.left_on else None,
        right_on=args.right_on.split(',') if args.right_on else None,
        suffixes=suffixes
    )

    if not args.quiet:
        print_join_stats(stats)

    save(merged_df, args.output)


if __name__ == '__main__':
    main()

