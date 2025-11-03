#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Spindle - Produce a spindle plot')
    parser.add_argument('subject', help='Path to dataset file (TSV)')
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV)')
    parser.add_argument('--group-col', required=True, help='Column in metadata to group by')
    parser.add_argument('--x', help='Column to use for x-axis (default: first column)')
    parser.add_argument('--y', help='Column to use for y-axis (default: second column)')
    parser.add_argument('--figsize', nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'),
                        default=(3, 3), help='Figure size in inches, e.g. --figsize 3 3')
    parser.add_argument('-o', '--output', help='Output image filename (default: spindle.svg)')
    return parser.parse_args()

def load_tsv(path):
    return pd.read_csv(path, sep='\t', index_col=0)

def merge_data(subject_df, meta_df, group_col):
    if group_col not in meta_df.columns:
        raise ValueError(f"Metadata missing column '{group_col}'")
    merged = subject_df.join(meta_df[group_col], how='inner').dropna(subset=[group_col])
    return merged.set_index(group_col)

def spindle(df, x=None, y=None, figsize=(3, 3)):
    if x is None or y is None:
        x, y = df.columns[:2]
    groups = df.index.unique()
    palette = dict(zip(groups, sns.color_palette("hls", len(groups)).as_hex()))

    centers = df.groupby(df.index)[[x, y]].mean().rename(columns={x: "mx", y: "my"})
    j = df.join(centers)

    plt.figure(figsize=figsize)
    for grp, sub in j.groupby(j.index):
        plt.plot([sub[x], sub["mx"]], [sub[y], sub["my"]], linewidth=0.5, color=palette[grp], alpha=0.3)
        plt.scatter(sub[x], sub[y], color=palette[grp], s=1)
        plt.scatter(sub["mx"].iloc[0], sub["my"].iloc[0], c='black', s=10, marker='+')
        plt.text(sub["mx"].iloc[0] + 0.002, sub["my"].iloc[0] + 0.002, str(grp))

    plt.xlabel(x)
    plt.ylabel(y)
    sns.despine()

def main():
    args = parse_arguments()
    df = load_tsv(args.subject)
    meta = load_tsv(args.meta)

    df = merge_data(df, meta, args.group_col)

    spindle(df)

    output = args.output or "spindle.svg"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved to {output}")

if __name__ == '__main__':
    main()

