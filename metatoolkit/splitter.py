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
    if df.empty:
        print(f"Skipping empty dataframe for '{folder}' — no file created.")
        return
    target_dir = os.path.join(output_dir, str(folder))
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, f'{filename}.tsv')
    df.to_csv(output_path, sep='\t', index=index)
    print(f"Saved split: {output_path}")


def splitter(df, df2, column, reindex_col=None, drop_index=False):
    """Split df into sub-dataframes based on unique values in df2[column] (or df if df2 not provided)."""
    output = {}

    # Use df2 if provided, else use df directly
    meta = df2 if df2 is not None else df

    if column not in meta.columns:
        raise ValueError(f"Column '{column}' not found in metadata or input dataframe.")

    for level in meta[column].unique():
        subset_meta = meta[meta[column] == level]

        # If df2 is provided, perform join; otherwise just subset df directly
        if df2 is not None:
            merged = df.join(subset_meta[[column] + ([reindex_col] if reindex_col else [])], how='inner')
        else:
            merged = df.loc[subset_meta.index.intersection(df.index)].copy()

        # Remove split column if present
        if column in merged.columns:
            merged = merged.drop(columns=[column])

        # Reindex if specified
        if reindex_col:
            if reindex_col not in merged.columns:
                raise ValueError(f"Reindex column '{reindex_col}' not found in metadata or dataframe.")
            merged = merged.set_index(reindex_col)

        # Optionally drop index entirely
        if drop_index:
            merged.reset_index(drop=True, inplace=True)

        # Only include non-empty splits
        if not merged.empty:
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
    parser.add_argument('--drop-index', action='store_true',
                        help='Drop the index in output files after splitting')
    return parser.parse_args()




def main():
    args = parse_args()
    df = load(args.subject)
    df2 = load(args.df2) if args.df2 else None

    # If df2 is None, don’t attempt metadata-based operations
    output = splitter(df, df2, args.column, args.reindex, args.drop_index)
    base_filename = os.path.splitext(os.path.basename(args.subject))[0]
    output_dir = os.path.abspath(args.outdir)

    for level, split_df in output.items():
        save(split_df, output_dir, level, base_filename, index=not args.drop_index)

    print(f"Split files saved to subfolders in: {output_dir}")


if __name__ == '__main__':
    main()

