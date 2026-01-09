#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import sys

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="kmeans_cluster - Apply K-means clustering to your dataset"
    )

    parser.add_argument(
        "subject",
        help="Path to dataset file or subject name"
    )

    parser.add_argument(
        "--clusters",
        type=int,
        help="Force number of clusters (overrides Silhouette method)"
    )

    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum number of clusters to test (default: 2)"
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum number of clusters to test (default: 10)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for K-means initialization (default: 0)"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .tsv file path (directories will be created if needed)"
    )

    return parser.parse_args()


def load_subject_df(path_or_name):
    path = Path(path_or_name)
    if path.is_file():
        df = pd.read_csv(path, sep="\t", index_col=0)
        stem = path.stem
    else:
        df = pd.read_csv(f"{path_or_name}.tsv", sep="\t", index_col=0)
        stem = path_or_name

    return df, stem


def select_numeric(df):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        sys.exit("ERROR: no numeric columns found for clustering")
    return numeric


def find_optimal_k(data, min_k, max_k, random_state):
    if min_k >= max_k:
        sys.exit("ERROR: --min-clusters must be < --max-clusters")

    best_k = None
    best_score = -1

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(data)

        score = silhouette_score(data, labels)
        print(f"K={k}, Silhouette score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Selected optimal K={best_k} (Silhouette={best_score:.4f})")
    return best_k


def run_kmeans(df, args):
    numeric = select_numeric(df)

    # Default: automatic K selection
    if args.clusters is None:
        k = find_optimal_k(
            numeric,
            args.min_clusters,
            args.max_clusters,
            args.random_state
        )
    else:
        k = args.clusters
        print(f"Using user-specified K={k}")

    km = KMeans(n_clusters=k, random_state=args.random_state)
    labels = km.fit_predict(numeric)

    # Convert numeric labels to cluster_1, cluster_2, ...
    labels = pd.Series(
        [f"cluster_{i + 1}" for i in labels],
        index=df.index,
        name="Cluster"
    )

    return labels, k

def save_clusters(clusters, output_path):
    output_path = Path(output_path)

    if output_path.suffix != ".tsv":
        sys.exit("ERROR: --output must be a .tsv file path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output_path, sep="\t", header=True)

    print(f"Saved cluster labels to {output_path}")


def main():
    args = parse_arguments()
    df, _ = load_subject_df(args.subject)

    clusters, _ = run_kmeans(df, args)
    save_clusters(clusters, args.output)


if __name__ == "__main__":
    main()

