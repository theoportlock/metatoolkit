#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform PCoA on a distance edge list (source, target, distance).
"""

import argparse
import os
import pandas as pd
import skbio


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform PCoA on a distance edge list (source, target, distance)")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input edge list file (TSV) with columns: source, target, distance"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path (TSV)"
    )
    return parser.parse_args()


def load_edge_list(path: str) -> pd.DataFrame:
    """Load an edge list and convert it into a square distance matrix."""
    df = pd.read_csv(path, sep='\t')

    required_cols = {"source", "target", "distance"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Edge list must contain columns: source, target, distance")

    # Pivot to create square distance matrix
    dist_mat = df.pivot(index="source", columns="target", values="distance")

    return dist_mat


def perform_pcoa(df: pd.DataFrame) -> pd.DataFrame:
    """Perform PCoA and return the first two coordinates."""
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

    # Load edge list
    df = load_edge_list(args.input)

    # Perform PCoA
    result = perform_pcoa(df)

    # Save output
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    result.to_csv(args.output, sep='\t')
    print(f"PCoA results saved to: {args.output}")


if __name__ == "__main__":
    main()

