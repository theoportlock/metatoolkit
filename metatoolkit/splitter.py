#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd


def load(path):
    """Load TSV from absolute path."""
    return pd.read_csv(path, sep='\t', index_col=0)


def save(df, output_dir, folder, filename, index=True):
    """Save dataframe to output_dir/folder/filename.tsv"""
    target_dir = os.path.join(output_dir, str(folder))
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, f'{filename}.tsv')
    df.to_csv(output_path, sep='\t', index=index)


def splitter(df, df2, column, reindex_col=None):
    """Split df into sub-dataframes based on unique values in df2[column]."""
    output = {}
    if df2 is None:
        df2 = df.copy()
    for level in df2[column].unique():
        subset_meta = df2[df2[column] == level]

        merged = df.join(subset_meta[[column] + ([reindex_col] if reindex_col else [])], how='inner')

        # Remove split column
        merged = merged.drop(columns=[column])

        # Reindex if specified
        if reindex_col:
            if reindex_col not in merged.columns:
                raise ValueError(f"Reindex column '{reindex_col}' not found in metadata.")
            merged = merged.set_index(reindex_col)

        output[level] = merged
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='''
    Splitter - splits dataframes according to the values in a defined column
    and optionally reindexes using a column from metadata.
    ''')
    parser.add_argument('subject', help='Input TSV file path')
    parser.add_argument('-col', '--column', required=True, help='Column name to split by (in metadata)')
    parser.add_argument('-m', '--df2', help='Optional second TSV file for metadata')
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--reindex', help='Optional column name in metadata to use as new index')
    return parser.parse_args()


def main():
    args = parse_args()
    df = load(args.subject)
    df2 = load(args.df2) if args.df2 else None

    output = splitter(df, df2, args.column, args.reindex)
    base_filename = os.path.splitext(os.path.basename(args.subject))[0]
    output_dir = os.path.abspath(args.outdir)

    for level, split_df in output.items():
        save(split_df, output_dir, level, base_filename)

    print(f"Split files saved to subfolders in: {output_dir}")


if __name__ == '__main__':
    main()

