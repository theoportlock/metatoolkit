#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute and save distance matrices as an edgelist.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Compute distance edgelist from a data table')
    parser.add_argument(
        'subject',
        type=str,
        help='Path to input data file (TSV) or subject name to look up in results/'
    )
    parser.add_argument(
        '-m', '--metric',
        type=str,
        default='braycurtis',
        help='Distance metric (e.g., braycurtis, euclidean, cosine, etc.)'
    )
    parser.add_argument(
        '-o', '--outfile',
        type=str,
        required=True,
        help='Output file path (TSV). Required.'
    )
    return parser.parse_args()


def dist_edgelist(df: pd.DataFrame, metric: str = 'braycurtis') -> pd.DataFrame:
    """Compute pairwise distances and return as a full edgelist (includes duplicates and self-distances)."""
    dist_array = squareform(pdist(df.values, metric=metric))
    dist_df = pd.DataFrame(dist_array, index=df.index, columns=df.index)
    dist_df.index.name = 'source'
    dist_df.columns.name = 'target'
    edge_df = dist_df.stack().rename_axis(['source', 'target']).reset_index(name='distance')
    return edge_df


def load(subject: str) -> pd.DataFrame:
    """Load the data table from file or from results folder."""
    if not os.path.exists(subject):
        raise FileNotFoundError(f"Input file not found: {subject}")
    return pd.read_csv(subject, sep='\t', index_col=0)


def save(df: pd.DataFrame, outpath: str) -> None:
    """Save DataFrame to TSV, creating directories as needed."""
    outdir = os.path.dirname(outpath)
    if outdir:  # only create directories if path includes a folder
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(outpath, sep='\t', index=False)


def main():
    args = parse_args()

    # Determine subject name (for logging)
    subject_arg = args.subject
    if os.path.isfile(subject_arg):
        subject_name = Path(subject_arg).stem
    else:
        subject_name = subject_arg

    # Load data
    df = load(subject_arg)

    # Compute edgelist
    edge_df = dist_edgelist(df, metric=args.metric)

    # Save output
    save(edge_df, args.outfile)
    print(f"Distance edgelist saved to: {args.outfile}")
    print(f"Rows: {len(edge_df)} (includes duplicates and self-distances)")


if __name__ == '__main__':
    main()

