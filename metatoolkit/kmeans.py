#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import sys
from sklearn.cluster import KMeans

def parse_arguments():
    parser = argparse.ArgumentParser(description='''
    kmeans_cluster - Apply K-means clustering to your dataset
    ''')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument(
        '--clusters',
        type=int,
        default=3,
        help='Number of clusters for K-means (default: 3)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=0,
        help='Random state for K-means initialization (default: 0)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output filename (without extension). Default: {subject}_clusters'
    )
    return parser.parse_args()

def load_subject_df(path_or_name):
    """Load main data from ../results/{stem}.tsv"""
    path = Path(path_or_name)
    stem = path.stem if path.is_file() else path_or_name
    df = pd.read_csv(Path('../results') / f'{stem}.tsv', sep='\t', index_col=0)
    return df, stem

def run_kmeans(df, n_clusters, random_state):
    """Fit KMeans on all numeric columns and return cluster labels"""
    numeric = df.select_dtypes(include='number')
    if numeric.empty:
        sys.exit("ERROR: no numeric columns found for clustering")
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(numeric)
    return pd.Series(labels, index=df.index, name='Cluster')

def save_clusters(clusters, stem, out_name=None):
    """Save cluster labels to ../results/{stem}_clusters.tsv"""
    out_dir = Path('../results')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_name or f'{stem}_clusters'
    clusters.to_csv(out_dir / f'{out_stem}.tsv', sep='\t', header=True)
    print(f"Saved cluster labels to {out_dir / f'{out_stem}.tsv'}")

def main():
    args = parse_arguments()
    df, stem = load_subject_df(args.subject)
    clusters = run_kmeans(df, args.clusters, args.random_state)
    save_clusters(clusters, stem, args.output)

if __name__ == '__main__':
    main()
