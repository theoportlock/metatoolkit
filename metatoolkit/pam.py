#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import calinski_harabasz_score
from skbio.stats.ordination import pcoa


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="""
    pam_bc_ch - PAM clustering on Bray–Curtis distances with
    Calinski–Harabasz-based selection of k
    """)

    parser.add_argument(
        'subject',
        help='Distance file stem or path (results/{stem}.tsv)'
    )

    parser.add_argument(
        '--k-min',
        type=int,
        default=2,
        help='Minimum number of clusters to test (default: 2)'
    )

    parser.add_argument(
        '--k-max',
        type=int,
        default=20,
        help='Maximum number of clusters to test (default: 20)'
    )

    parser.add_argument(
        '--pcoa-components',
        type=int,
        default=None,
        help='Number of PCoA axes to retain (default: all)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output filename (without extension). Default: {subject}_clusters'
    )

    return parser.parse_args()


# --------------------------------------------------
# IO
# --------------------------------------------------

def load_distance_table(path_or_name):
    path = Path(path_or_name)
    stem = path.stem if path.is_file() else path_or_name

    df = pd.read_csv(
        Path('results') / f'{stem}.tsv',
        sep='\t'
    )

    required = {'source', 'target', 'distance'}
    if not required.issubset(df.columns):
        sys.exit(
            "ERROR: distance file must contain columns: "
            "source, target, distance"
        )

    return df, stem


def long_to_square(df):
    samples = pd.Index(
        sorted(set(df['source']) | set(df['target']))
    )

    D = pd.DataFrame(
        np.zeros((len(samples), len(samples))),
        index=samples,
        columns=samples,
        dtype=float
    )

    for s, t, d in df[['source', 'target', 'distance']].itertuples(index=False):
        D.loc[s, t] = d
        D.loc[t, s] = d

    return D


# --------------------------------------------------
# Core methods
# --------------------------------------------------

def pam_fit(D, k):
    model = KMedoids(
        n_clusters=k,
        metric='precomputed',
        method='pam',
        init='k-medoids++',
        random_state=0
    )
    model.fit(D)
    return model


def braycurtis_pcoa(D, n_components=None):
    ord_res = pcoa(D.values)
    coords = ord_res.samples.values

    if n_components is not None:
        coords = coords[:, :n_components]

    return coords


def ch_index_k_search(D, k_min, k_max, n_components=None):
    coords = braycurtis_pcoa(D, n_components)

    scores = {}

    for k in range(k_min, k_max + 1):
        model = pam_fit(D, k)
        labels = model.labels_

        if len(np.unique(labels)) < 2:
            continue

        ch = calinski_harabasz_score(coords, labels)
        scores[k] = ch

    if not scores:
        sys.exit("ERROR: Calinski–Harabasz index could not be computed")

    best_k = max(scores, key=scores.get)
    return best_k, scores


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def run_pam_bc_ch(df_long, args):
    D = long_to_square(df_long)

    best_k, ch_scores = ch_index_k_search(
        D,
        args.k_min,
        args.k_max,
        args.pcoa_components
    )

    print(f"Selected k={best_k} via Calinski–Harabasz index")

    final_model = pam_fit(D, best_k)

    clusters = pd.Series(
        final_model.labels_,
        index=D.index,
        name='Cluster'
    )

    return clusters, final_model, ch_scores


# --------------------------------------------------
# Output
# --------------------------------------------------

def save_clusters(clusters, stem, out_name=None):
    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stem = out_name or f'{stem}_clusters'
    out_path = out_dir / f'{out_stem}.tsv'

    clusters.to_csv(out_path, sep='\t', header=True)
    print(f"Saved cluster labels to {out_path}")


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

def main():
    args = parse_arguments()
    df_long, stem = load_distance_table(args.subject)

    clusters, model, ch_scores = run_pam_bc_ch(df_long, args)
    save_clusters(clusters, stem, args.output)

    print(f"Final inertia: {model.inertia_}")
    print(f"Medoid sample IDs: {list(clusters.index[model.medoid_indices_])}")


if __name__ == '__main__':
    main()

