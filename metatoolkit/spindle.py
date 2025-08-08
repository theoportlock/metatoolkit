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

def load_subject_df(path):
    """Load main data from results/{stem}.tsv"""
    path = Path(path)
    stem = path.stem
    df = pd.read_csv(path, sep='\t', index_col=0)
    return df, stem

def load_and_merge_meta(df, meta_path, group_col):
    """Load metadata, check group_col exists, set it as the index, then inner-join to df."""
    # auto-detect delimiter
    mdf = pd.read_csv(meta_path, sep='\t', index_col=0)
    if group_col not in mdf.columns:
        sys.exit(f"ERROR: metadata file {meta_path} does not contain column '{group_col}'")
    return df.join(mdf, how='inner').set_index(group_col)

def spindle(df, ax=None, palette=None):
    """Draw a spindle plot grouping by df.index."""
    # build palette if none supplied
    groups = df.index.unique()
    if palette is None:
        palette = pd.Series(
            sns.color_palette("hls", len(groups)).as_hex(),
            index=groups
        )

    if ax is None:
        fig, ax = plt.subplots()

    x, y = df.columns[:2]

    # compute group centers
    centers = df.groupby(df.index)[[x, y]].mean()
    centers.columns = ['nPC1', 'nPC2']

    # join centers back onto each row
    j = df.join(centers, how='inner')
    j['colour'] = j.index.map(palette)

    # draw spokes + points
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

    # draw and label centers
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
    plt.savefig(out / f'{stem}.svg')
    plt.clf()

def main():
    args = parse_arguments()

    # load subject data
    df, stem = load_subject_df(args.subject)

    # load metadata, ensure group-col present, set as index, then join
    df = load_and_merge_meta(df, args.meta, args.group_col)

    # now df.index holds your explainer groups
    ax = spindle(df)

    savefig(f'{stem}_spindle')

if __name__ == '__main__':
    main()
