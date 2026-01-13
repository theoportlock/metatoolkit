#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute summary statistics of taxonomic presence by rank.

Presence is defined as abundance > presence_threshold.

Output includes:
- total_features : total features detected at each rank
- mean_count     : mean number of present features per sample
- median_count   : median number of present features per sample
- iqr_count      : interquartile range of present features per sample
- std_count      : standard deviation of present features per sample

Ranks included:
Kingdom, Phylum, Class, Order, Family, Genus, Species, SGB
"""

from pathlib import Path
import argparse
import pandas as pd

# Full biological rank order including separate SGB row
RANK_ORDER = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "SGB"]

# Map column initials to full rank names
RANK_MAP = {
    "K": "Kingdom",
    "P": "Phylum",
    "C": "Class",
    "O": "Order",
    "F": "Family",
    "G": "Genus",
    "S": "Species",
    "T": "SGB",       # T = species-level genome bins (separate category)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a taxonomic abundance table by rank-level presence."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input taxonomy abundance TSV file (rows=samples, columns=features)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output TSV file",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum abundance to consider a feature present "
            "(default: >0.0)"
        ),
    )
    return parser.parse_args()


def extract_rank(columns: pd.Index) -> pd.Series:
    """
    Extract taxonomic rank initial from feature names and map to full rank names.
    """
    rank_initial = columns.str.rsplit("|", n=1).str[-1].str[0].str.upper()
    return rank_initial.map(RANK_MAP)


def taxo_summary(
    df: pd.DataFrame,
    presence_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute per-rank summary statistics.

    Parameters
    ----------
    df : DataFrame
        Rows = samples, columns = taxonomic features
    presence_threshold : float
        Minimum abundance to count a feature as present

    Returns
    -------
    DataFrame
        Per-rank summary statistics
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    # Map features to full rank names
    ranks = extract_rank(df.columns)

    # Total number of features per rank
    total_features = ranks.value_counts().rename("total_features")

    # Presence/absence matrix
    presence = (df > presence_threshold).astype(int)

    # Per-sample counts by rank
    per_sample = presence.groupby(ranks, axis=1).sum()

    # IQR calculation
    iqr = (per_sample.quantile(0.75) - per_sample.quantile(0.25)).rename("iqr_count")

    # Aggregate statistics
    summary = pd.concat(
        [
            total_features,
            per_sample.mean().rename("mean_count"),
            per_sample.median().rename("median_count"),
            iqr,
            per_sample.std().rename("std_count"),
        ],
        axis=1,
    )

    # Reindex to enforce biological order including SGB
    summary = summary.reindex(RANK_ORDER)

    return summary


def main():
    args = parse_args()

    # Read input TSV
    df = pd.read_csv(args.input, sep="\t", index_col=0)

    # Compute summary
    summary = taxo_summary(df, presence_threshold=args.presence_threshold)

    # Write output TSV
    summary.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()

