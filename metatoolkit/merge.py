#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    df.to_csv(subject, sep='\t', index=index)

def merge(datasets=None, join='inner', append=False, add_filename_column=False, filenames=None):
    if datasets is None:
        raise ValueError("datasets argument must not be None")
    if join not in ('inner', 'outer'):
        raise ValueError("join argument must be either 'inner' or 'outer'")
    if append:
        if add_filename_column and filenames is not None:
            for df, fname in zip(datasets, filenames):
                df['filename'] = os.path.basename(fname)
        return pd.concat(datasets, axis=0, join=join)
    return pd.concat(datasets, axis=1, join=join)

def parse_args():
    parser = argparse.ArgumentParser(description='Merge - Combines datasets')
    parser.add_argument('datasets', nargs='+', help='List of dataset files to merge')
    parser.add_argument('-j', '--join', default='inner', help='Join method: inner, outer, etc.')
    parser.add_argument('-a', '--append', action='store_true', help='Append vertically instead of joining horizontally')
    parser.add_argument('--add-filename', action='store_true', help='When appending, add a column with the source filename')
    parser.add_argument('-o', '--output', help='Name of the output file')
    return parser.parse_args()

def main():
    args = parse_args()
    dfs = [load(df) for df in args.datasets]
    result = merge(
        datasets=dfs, 
        join=args.join, 
        append=args.append, 
        add_filename_column=args.add_filename, 
        filenames=args.datasets
    )

    print(result)

    output_name = args.output if args.output else f'results/{"_".join(args.datasets)}'
    save(result, output_name)

if __name__ == '__main__':
    main()
