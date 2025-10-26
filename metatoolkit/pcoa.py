#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform PCoA on a distance edge list (source, target, distance).
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
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input edge list file (TSV) with columns: source, target, DIST"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path (TSV)"
    )
    parser.add_argument(
        "-d", "--distance",
        type=str,
        default="bray-curtis",
        help="Distance column to use (default: bray-curtis)"
    )
    return parser.parse_args()


def load_edge_list(path: str, dist_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')

    required_cols = {"source", "target", dist_col}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Pivot to square matrix
    dist_mat = df.pivot(index="source", columns="target", values=dist_col)

    # Symmetrize (use whichever triangle exists)
    dist_mat = dist_mat.combine_first(dist_mat.T)

    # Fill diagonal
    for i in dist_mat.index:
        dist_mat.loc[i, i] = 0.0

    # Remaining missing values get replaced with max observed distance
    if dist_mat.isna().values.any():
        max_dist = np.nanmax(dist_mat.values)
        dist_mat = dist_mat.fillna(max_dist)
        print(f"Warning: Missing values filled with max distance {max_dist:.4f}")

    return dist_mat


def perform_pcoa(df: pd.DataFrame) -> pd.DataFrame:
    DM = skbio.stats.distance.DistanceMatrix(df.values, ids=df.index)
    pcoa_res = skbio.stats.ordination.pcoa(DM, number_of_dimensions=2)

    explained = pcoa_res.proportion_explained
    result = pd.DataFrame({
        f"PCo1 ({explained['PC1']:.1%})": pcoa_res.samples.iloc[:, 0],
        f"PCo2 ({explained['PC2']:.1%})": pcoa_res.samples.iloc[:, 1],
    }, index=df.index)

    return result


def main():
    args = parse_arguments()

    dist_mat = load_edge_list(args.input, args.distance)
    result = perform_pcoa(dist_mat)

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result.to_csv(args.output, sep='\t')
    print(f"PCoA results saved to: {args.output}")


if __name__ == "__main__":
    main()

