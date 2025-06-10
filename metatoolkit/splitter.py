#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd


def load(subject):
    """Load TSV from absolute path."""
    return pd.read_csv(subject, sep='\t', index_col=0)


def save(df, output_dir, filename, index=True):
    """Save dataframe to absolute output path inside output_dir/filename.tsv"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{filename}.tsv')
    df.to_csv(output_path, sep='\t', index=index)


def splitter(df, df2, column):
    """Split df into sub-dataframes based on unique values in df2[column]."""
    output = {}
    if df2 is None:
        df2 = df.copy()
    for level in df2[column].unique():
        merged = df.loc[:, df.columns[df.columns != column]].join(
            df2.loc[df2[column] == level, column], how='inner'
        ).drop(columns=[column])
        output[level] = merged
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='''
    Splitter - splits dataframes according to the values in a defined column''')
    parser.add_argument('subject', help='TSV file path')
    parser.add_argument('column', help='Column name to split by')
    parser.add_argument('--df2', help='Optional second TSV file or basename')
    parser.add_argument('--outdir', help='Optional output directory (absolute or relative)')
    return parser.parse_args()


def main():
    args = parse_args()
    subject = args.subject
    column = args.column

    df = load(subject)
    df2 = load(args.df2) if args.df2 else None

    output = splitter(df, df2, column)

    output_dir = os.path.abspath(args.outdir)

    for level, split_df in output.items():
        save(split_df, output_dir, level)

    print(f"✅ Split files saved to: {output_dir}")


if __name__ == '__main__':
    main()

