#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic pastel Seaborn pointplot with subject-level overlay."
    )
    parser.add_argument("subject", help="Path to subject-level data TSV file")
    parser.add_argument("-m", "--meta", required=True, help="Path to metadata TSV file")
    parser.add_argument("-x", required=True, help="X-axis variable (e.g., timepoint)")
    parser.add_argument("-y", required=True, help="Y-axis variable (e.g., WLZ_WHZ)")
    parser.add_argument("--hue", required=True, help="Grouping variable for color (e.g., Feed)")
    parser.add_argument("--id", required=True, help="Unique subject ID column name (e.g., subjectID)")
    parser.add_argument("--logy", action="store_true", help="Use log scale for Y-axis")
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file path")
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[3, 3], metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches, e.g., --figsize 4 4"
    )
    return parser.parse_args()


def plot(df, x, y, hue, id_col, output, logy=False, figsize=(3, 3)):
    # Drop missing values in relevant columns
    df = df.dropna(subset=[x, y, hue, id_col])

    # Define pastel color palette
    hue_values = df[hue].unique()
    pastel_palette = sns.color_palette("pastel", len(hue_values))
    palette = dict(zip(hue_values, pastel_palette))

    fig, ax = plt.subplots(figsize=figsize)

    # Add individual points with light transparency (for subject-level info)
    sns.stripplot(
        data=df, x=x, y=y, hue=hue, palette=palette,
        alpha=0.5, jitter=True, size=2, ax=ax, legend=False
    )

    # Add group means and CIs
    sns.pointplot(
        data=df, x=x, y=y, hue=hue, palette=palette,
        join=True, markers="o", scale=0.8,
        errwidth=1, linewidth=1, ax=ax, errorbar="se"
    )

    if logy:
        ax.set_yscale("log")

    ax.set_title(y)
    ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output)
    print(f"Plot saved to {output}")


def main():
    args = parse_args()

    # Load and merge data
    df_subject = pd.read_csv(args.subject, sep="\t", index_col=0)
    df_meta = pd.read_csv(args.meta, sep="\t", index_col=0)
    df = df_subject.join(df_meta)

    plot(df, args.x, args.y, args.hue, args.id, args.output, args.logy, tuple(args.figsize))


if __name__ == "__main__":
    main()

