#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='''
    Spindle - Produces a spindleplot of a given dataset
    ''')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument(
        '--meta',
        help='Path to a metadata file (TSV/CSV) containing your explainer column',
        required=True
    )
    parser.add_argument(
        '--group-col',
        help='Name of the column in the metadata file to use for grouping/spindle labels',
        required=True
    )
    return parser.parse_args()

def load_subject_df(path_or_name):
    """Load main data from results/{stem}.tsv"""
    path = Path(path_or_name)
    stem = path.stem if path.is_file() else path_or_name
    df = pd.read_csv(Path('results') / f'{stem}.tsv', sep='\t', index_col=0)
    df.index = df.index.astype(str)  # Force index to string
    return df, stem

def load_and_merge_meta(df, meta_path, group_col):
    """Load metadata and assign group labels from group_col."""
    # auto-detect delimiter, load with default index
    mdf = pd.read_csv(meta_path, sep=None, engine='python', index_col=0)
    mdf.index = mdf.index.astype(str)  # Force index to string

    if group_col not in mdf.columns:
        sys.exit(f"ERROR: metadata file {meta_path} does not contain column '{group_col}'")

    if not df.index.equals(mdf.index):
        print("WARNING: Index mismatch between subject and metadata. Trying inner join.")
        df = df.join(mdf[[group_col]], how='inner')
    else:
        df[group_col] = mdf[group_col]

    df = df.dropna(subset=[group_col])  # remove any rows missing group info
    df = df.set_index(group_col)        # group_col becomes the new index
    return df

def spindle(df, ax=None, palette=None):
    """Draw a spindle plot grouping by df.index."""
    groups = df.index.unique()
    if palette is None:
        palette = pd.Series(
            sns.color_palette("hls", len(groups)).as_hex(),
            index=groups
        )

    if ax is None:
        fig, ax = plt.subplots()

    x, y = df.columns[:2]

    centers = df.groupby(df.index)[[x, y]].mean()
    centers.columns = ['nPC1', 'nPC2']

    j = df.join(centers, how='inner')
    j['colour'] = j.index.map(palette)

    for _, row in j.iterrows():
        ax.plot(
            [row[x], row['nPC1']],
            [row[y], row['nPC2']],
            linewidth=0.5,
            color=row['colour'],
            alpha=0.3,
            zorder=1
        )
        ax.scatter(row[x], row[y], color=row['colour'], s=1, zorder=1)

    for grp, cen in centers.iterrows():
        ax.scatter(cen['nPC1'], cen['nPC2'], c='black', s=10, marker='+', zorder=2)
        ax.text(cen['nPC1'] + 0.002, cen['nPC2'] + 0.002, str(grp), zorder=3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def savefig(stem):
    out = Path('results')
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f'{stem}_spindle.svg')
    plt.clf()

def main():
    args = parse_arguments()

    df, stem = load_subject_df(args.subject)
    df = load_and_merge_meta(df, args.meta, args.group_col)

    if df.empty:
        sys.exit("ERROR: No data left after merging subject and metadata. Check for index mismatch.")

    ax = spindle(df)
    savefig(stem)

if __name__ == '__main__':
    main()
