#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_abundance(
    df: pd.DataFrame,
    order: bool,
    max_categories: int,
    figsize: tuple,
    normalize: bool,
    log_transform: bool
):
    # Apply log1p transformation if requested
    if log_transform:
        # We apply log1p to the raw values before normalization/stacking
        df = np.log1p(df)

    # Keep top N columns + combine others
    col_means = df.mean()
    sorted_cols = col_means.sort_values(ascending=False)

    if len(sorted_cols) > max_categories:
        top_cols = sorted_cols.iloc[:max_categories].index
        other_cols = sorted_cols.iloc[max_categories:].index
        df["others"] = df[other_cols].sum(axis=1)
        df = df[top_cols.tolist() + ['others']]

    if normalize:
        # Avoid division by zero if a row is all zeros
        row_sums = df.sum(axis=1)
        df = df.div(row_sums.replace(0, 1), axis=0)

    if order:
        # Sort the columns (stacks) by their mean abundance for a cleaner look
        df = df[df.mean().sort_values().index]

    # Plot
    ax = df.plot(kind="bar", stacked=True, figsize=figsize, width=0.9, cmap="tab20")

    ylabel = "Abundance"
    if log_transform:
        ylabel = "log1p(Abundance)"
    if normalize:
        ylabel = f"Relative {ylabel}"
        ax.set_ylim(0, 1)

    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return plt

def main():
    parser = argparse.ArgumentParser(description="Plot stacked bar chart of abundances from a TSV file.")

    # Positional argument
    parser.add_argument("input", type=Path, help="TSV file with samples as rows, features as columns")

    # Optional arguments
    parser.add_argument("-o", "--output", type=Path, default="abund.svg", help="Output file path (default: abund.svg)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[4.0, 4.0], help="Figure size: width height (default: 4.0 4.0)")
    parser.add_argument("--order", action="store_true", help="Sort categories by average abundance")
    parser.add_argument("--max-categories", type=int, default=20, help="Top categories to keep (default: 20)")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize", help="Do not normalize rows to relative abundance")
    parser.add_argument("--log1p", action="store_true", help="Apply log1p transformation to abundances")

    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    # Validation
    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        return

    # Load data
    try:
        df = pd.read_csv(args.input, sep="\t", index_col=0)
    except Exception as e:
        print(f"❌ Error reading TSV: {e}")
        return

    # Generate plot
    plot = plot_abundance(
        df,
        order=args.order,
        max_categories=args.max_categories,
        figsize=tuple(args.figsize),
        normalize=args.normalize,
        log_transform=args.log1p
    )

    plot.tight_layout()
    plot.savefig(args.output)
    print(f"✅ Saved plot to: {args.output}")

if __name__ == "__main__":
    main()
