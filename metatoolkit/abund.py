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
    log_transform: bool,
):

    if log_transform:
        df = np.log1p(df)

    col_means = df.mean()
    sorted_cols = col_means.sort_values(ascending=False)

    if len(sorted_cols) > max_categories:
        top_cols = sorted_cols.iloc[:max_categories].index
        other_cols = sorted_cols.iloc[max_categories:].index

        df = pd.concat(
            [df[top_cols], df[other_cols].sum(axis=1).rename("others")],
            axis=1,
        )

    if normalize:
        row_sums = df.sum(axis=1)
        df = df.div(row_sums.replace(0, 1), axis=0)

    if order:
        df = df[df.mean().sort_values().index]

    ax = df.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        width=0.9,
        cmap="tab20"
    )

    ylabel = "Abundance"

    if log_transform:
        ylabel = "log1p(Abundance)"

    if normalize:
        ylabel = f"Relative {ylabel}"
        ax.set_ylim(0, 1)

    ax.set_ylabel(ylabel)

    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize="small"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def main():

    parser = argparse.ArgumentParser(
        description="Plot stacked bar chart of abundances or species prevalence from a TSV file."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="TSV file with samples as rows, features as columns"
    )

    parser.add_argument(
        "--meta",
        type=Path,
        help="Optional metadata table to merge by index"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="abund.svg",
        help="Output file path"
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[4.0, 4.0],
        help="Figure size: width height"
    )

    parser.add_argument(
        "--order",
        action="store_true",
        help="Sort stacked categories by mean abundance"
    )

    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Top categories to keep"
    )

    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Do not normalize rows to relative abundance"
    )

    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Apply log1p transformation"
    )

    parser.add_argument(
        "--species",
        type=str,
        help="Species column to plot prevalence for"
    )

    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=["timepoint", "arm"],
        help="Metadata columns defining bar groups"
    )

    parser.add_argument(
        "--category-order",
        nargs="+",
        help="Explicit ordering of x-axis categories"
    )

    parser.add_argument(
        "--y-max",
        type=float,
        help="Set maximum y-axis value"
    )

    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}")
        return

    try:
        df = pd.read_csv(args.input, sep="\t", index_col=0)
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return

    # Merge metadata
    if args.meta:

        if not args.meta.exists():
            print(f"Metadata file not found: {args.meta}")
            return

        try:
            meta = pd.read_csv(args.meta, sep="\t", index_col=0)
        except Exception as e:
            print(f"Error reading metadata TSV: {e}")
            return

        df = df.join(meta, how="left").copy()

    # Species prevalence mode
    if args.species:

        df_reset = df.reset_index()

        if args.species not in df_reset.columns:
            raise ValueError(f"Species column not found: {args.species}")

        missing = [c for c in args.group_cols if c not in df_reset.columns]
        if missing:
            raise ValueError(f"Missing grouping columns: {missing}")

        prev = pd.to_numeric(df_reset[args.species], errors="coerce")

        groups = df_reset[args.group_cols].astype(str).agg(" ".join, axis=1)

        tmp = pd.DataFrame({
            "group": groups,
            "present": prev
        })

        tmp = tmp.groupby("group", as_index=True)["present"].mean()

        df = pd.DataFrame({
            "present": tmp,
            "absent": 1 - tmp
        })

    # Apply category ordering
    if args.category_order:
        df = df.reindex(args.category_order)

    ax = plot_abundance(
        df,
        order=args.order,
        max_categories=args.max_categories,
        figsize=tuple(args.figsize),
        normalize=args.normalize,
        log_transform=args.log1p,
    )

    # Prevalence plots default to 0–1
    if args.species and args.y_max is None:
        ax.set_ylim(0, 1)

    if args.y_max is not None:
        ax.set_ylim(0, args.y_max)

    plt.tight_layout()
    plt.savefig(args.output)

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()

