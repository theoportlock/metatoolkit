#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a histogram from a TSV dataset.

Supports bin control, log scaling, KDE overlay, faceting,
metadata joins, and publication-friendly themes.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce a histogram plot from a TSV dataset."
    )
    parser.add_argument(
        "subject",
        help="Subject name or path to TSV file"
    )
    parser.add_argument(
        "-c", "--column",
        default="sig",
        help="Column to plot (default: sig)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        help="Number of histogram bins"
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply log10 scale to x-axis"
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="Overlay KDE on histogram"
    )
    parser.add_argument(
        "--facet",
        help="Column name to facet by (can come from metadata)"
    )
    parser.add_argument(
        "--meta",
        nargs="+",
        help="Path(s) to metadata TSV file(s) to inner-join before plotting"
    )
    parser.add_argument(
        "--theme",
        choices=["paper", "talk"],
        default="paper",
        help="Plot theme (default: paper)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output image file (default: results/<subject>_hist.svg)"
    )
    return parser.parse_args()


# -----------------------------
# Styling
# -----------------------------
def set_theme(theme):
    if theme == "paper":
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    elif theme == "talk":
        sns.set_theme(style="whitegrid", context="talk", font_scale=1.3)


# -----------------------------
# Data loading / merging
# -----------------------------
def load_data(path_or_name):
    path = Path(path_or_name)
    return pd.read_csv(path, sep="\t", index_col=0)


def merge_meta(df, meta_paths):
    """
    Inner-join one or more metadata tables onto df using the index.
    """
    for meta in meta_paths:
        meta_df = pd.read_csv(meta, sep="\t", index_col=0)
        df = df.join(meta_df, how="inner")
    return df


# -----------------------------
# Plotting
# -----------------------------
def plot_histogram(
    df,
    column,
    bins=None,
    kde=False,
    log_scale=False,
    facet=None
):
    if facet:
        g = sns.displot(
            data=df.reset_index(),
            x=column,
            col=facet,
            bins=bins,
            kde=kde,
            facet_kws={"sharex": True, "sharey": True}
        )

        if log_scale:
            g.set(xscale="log")

        g.set_axis_labels(column, "Count")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(f"Histogram of {column}")

        return g.fig

    else:
        fig, ax = plt.subplots(figsize=(6, 4))

        hist_kwargs = {
            "data": df,
            "x": column,
            "kde": kde,
            "ax": ax,
        }

        if bins is not None:
            hist_kwargs["bins"] = bins

        sns.histplot(**hist_kwargs)


        if log_scale:
            ax.set_xscale("log")

        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {column}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        return fig


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_theme(args.theme)

    # Resolve input
    subject_path = Path(args.subject)
    if subject_path.is_file():
        subject = subject_path.stem
        input_file = subject_path
    else:
        subject = args.subject
        input_file = Path("results") / f"{subject}.tsv"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    df = load_data(input_file)

    # Merge metadata if provided
    if args.meta:
        df = merge_meta(df, args.meta)

    # Output path
    output_file = (
        Path(args.output)
        if args.output
        else Path("results") / f"{subject}_hist.svg"
    )

    # Plot
    fig = plot_histogram(
        df=df,
        column=args.column,
        bins=args.bins,
        kde=args.kde,
        log_scale=args.log_scale,
        facet=args.facet
    )

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Histogram saved to: {output_file}")


if __name__ == "__main__":
    main()

