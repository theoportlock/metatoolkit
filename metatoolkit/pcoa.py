#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform PCoA on a distance edge list.
"""

import argparse
import os
import pandas as pd
import skbio
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform PCoA on a distance edge list (source, target, distance)"
    )

    # Positional input file
    parser.add_argument(
        "input",
        type=str,
        help="Input edge list file (TSV)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path (TSV)"
    )

    parser.add_argument(
        "--source-col",
        default="source",
        help="Column name for source node (default: source)"
    )

    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name for target node (default: target)"
    )

    parser.add_argument(
        "--distance-col",
        default="bray-curtis",
        help="Column name for distance value (default: bray-curtis)"
    )

    parser.add_argument(
        "--dims",
        type=int,
        default=2,
        help="Number of PCoA dimensions to compute (default: 2)"
    )

    return parser.parse_args()


def load_edge_list(
    path: str,
    source_col: str,
    target_col: str,
    distance_col: str,
) -> pd.DataFrame:
    """Load a long-format edge list and return a symmetric distance matrix."""
    df = pd.read_csv(path, sep="\t")

    required_cols = {source_col, target_col, distance_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Edge list must contain columns: {', '.join(required_cols)}"
        )

    # Collect all unique nodes
    nodes = pd.unique(df[[source_col, target_col]].values.ravel())

    # Initialise square distance matrix
    dist_mat = pd.DataFrame(
        index=nodes,
        columns=nodes,
        dtype=float,
    )

    # Fill distances symmetrically
    for _, row in df.iterrows():
        i = row[source_col]
        j = row[target_col]
        d = row[distance_col]

        dist_mat.loc[i, j] = d
        dist_mat.loc[j, i] = d

    # Set diagonal to zero
    for n in nodes:
        dist_mat.loc[n, n] = 0.0

    # Final validation
    if dist_mat.isna().any().any():
        missing = dist_mat.isna().sum().sum()
        raise ValueError(
            f"Distance matrix contains {missing} missing values. "
            "Edge list does not define all pairwise distances."
        )

    return dist_mat


def perform_pcoa(df: pd.DataFrame, dims: int) -> pd.DataFrame:
    """Perform PCoA and return coordinates."""
    DM = skbio.stats.distance.DistanceMatrix(df.values, ids=df.index)
    pcoa_res = skbio.stats.ordination.pcoa(
        DM,
        number_of_dimensions=dims
    )

    explained = pcoa_res.proportion_explained

    cols = {}
    for i in range(dims):
        pc = f"PC{i + 1}"
        label = f"PCo{i + 1} ({explained[pc]:.1%})"
        cols[label] = pcoa_res.samples.iloc[:, i]

    result = pd.DataFrame(cols, index=df.index)
    return result


def main():
    args = parse_arguments()

    # Load edge list
    df = load_edge_list(
        args.input,
        args.source_col,
        args.target_col,
        args.distance_col,
    )

    # Perform PCoA
    result = perform_pcoa(df, args.dims)

    # Save output
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result.to_csv(args.output, sep="\t")
    print(f"PCoA results saved to: {args.output}")


if __name__ == "__main__":
    main()

