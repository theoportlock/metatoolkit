#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute and save distance matrices.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.spatial.distance import pdist, squareform


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Compute distance matrix from a data table')
    parser.add_argument('subject', type=str,
                        help='Path to input data file (TSV) or subject name to look up in results')
    parser.add_argument('-m', '--metric', type=str, default='braycurtis',
                        help='Distance metric (e.g., braycurtis, euclidean, cosine, etc.)')
    parser.add_argument('-o', '--outfile', type=str,
                        help='Output file path (TSV). If not provided, saved to results/<subject><suffix>.tsv')
    parser.add_argument('-s', '--suffix', type=str,
                        help='Suffix to append to subject name for output filename when --outfile not given')
    return parser.parse_args()


def dist(df: pd.DataFrame, metric: str = 'braycurtis') -> pd.DataFrame:
    """Compute a square distance matrix from input DataFrame."""
    arr = squareform(pdist(df.values, metric=metric))
    return pd.DataFrame(arr, index=df.index, columns=df.index)


def load(subject: str) -> pd.DataFrame:
    """Load the data table from file or from results folder."""
    if os.path.isfile(subject):
        path = subject
    else:
        path = f'results/{subject}.tsv'
    return pd.read_csv(path, sep='\t', index_col=0)


def save(df: pd.DataFrame, outpath: str, index: bool = True) -> None:
    """Save DataFrame to TSV, creating directories as needed."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, sep='\t', index=index)


def main():
    args = parse_args()

    # Determine base subject name
    subject_arg = args.subject
    if os.path.isfile(subject_arg):
        subject_name = Path(subject_arg).stem
    else:
        subject_name = subject_arg

    # Load data
    df = load(subject_arg)

    # Compute distances
    dist_df = dist(df, metric=args.metric)
    print(dist_df)

    # Determine output path
    if args.outfile:
        outpath = args.outfile
    else:
        outpath = f'results/{subject_name}{args.suffix}.tsv'

    # Save
    save(dist_df, outpath)


if __name__ == '__main__':
    main()
