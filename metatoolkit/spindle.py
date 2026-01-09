#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def parse_arguments():
    parser = argparse.ArgumentParser(description='Spindle - Produce a spindle plot')
    parser.add_argument('subject', help='Path to dataset file (TSV)')
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV)')
    parser.add_argument('--group-col', required=True, help='Column in metadata to group by')
    parser.add_argument('--x', help='Column to use for x-axis (default: first column)')
    parser.add_argument('--y', help='Column to use for y-axis (default: second column)')
    parser.add_argument('--figsize', nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'),
                        default=(3, 3), help='Figure size in inches')
    parser.add_argument('-o', '--output', help='Output image filename (default: spindle.svg)')

    parser.add_argument('--ellipses', action='store_true',
                        help='Draw confidence ellipses per group')
    parser.add_argument('--ellipse-level', type=float, default=0.95,
                        help='Confidence level for ellipses (default: 0.95)')

    parser.add_argument('--palette', default='hls',
                        help='Seaborn palette name or comma-separated list of colors')
    parser.add_argument('--group-order',
                        help='Comma-separated group order (default: alphabetical)')
    parser.add_argument('--title-from-subject', action='store_true',
                        help='Use basename of subject file as figure title')


    return parser.parse_args()

def load_tsv(path):
    return pd.read_csv(path, sep='\t', index_col=0)

def merge_data(subject_df, meta_df, group_col):
    if group_col not in meta_df.columns:
        raise ValueError(f"Metadata missing column '{group_col}'")
    merged = subject_df.join(meta_df[group_col], how='inner').dropna(subset=[group_col])
    return merged.set_index(group_col)

def parse_palette(palette, groups):
    if ',' in palette:
        colors = palette.split(',')
        if len(colors) != len(groups):
            raise ValueError("Number of colors must match number of groups")
        return dict(zip(groups, colors))
    else:
        return dict(zip(groups, sns.color_palette(palette, len(groups)).as_hex()))

def confidence_ellipse(x, y, ax, level=0.95, **kwargs):
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(level, df=2))

    width, height = 2 * scale * np.sqrt(vals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)

def spindle(df, x=None, y=None, figsize=(3, 3),
            ellipses=False, ellipse_level=0.95,
            palette='hls', group_order=None):

    if x is None or y is None:
        x, y = df.columns[:2]

    groups = sorted(df.index.unique())
    if group_order:
        groups = group_order.split(',')

    palette = parse_palette(palette, groups)

    centers = (
        df.groupby(df.index)[[x, y]]
        .mean()
        .rename(columns={x: "mx", y: "my"})
    )
    j = df.join(centers)

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for grp in groups:
        sub = j.loc[grp]
        color = palette[grp]

        ax.plot(
            [sub[x], sub["mx"]],
            [sub[y], sub["my"]],
            linewidth=0.4,
            color=color,
            alpha=0.3
        )

        ax.scatter(
            sub[x],
            sub[y],
            color=color,
            s=4,
            alpha=1
        )

        ax.scatter(
            sub["mx"].iloc[0],
            sub["my"].iloc[0],
            c='black',
            s=35,
            marker='+',
            linewidths=0.7,
            zorder=5,
        )


        if ellipses:
            confidence_ellipse(
                sub[x].values,
                sub[y].values,
                ax,
                level=ellipse_level,
                edgecolor=color,
                linewidth=0.4,
                alpha=0.5
            )

        legend_handles.append(
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                label=str(grp),
                markerfacecolor=color,
                markeredgecolor='none',
                markeredgewidth=0,
                markersize=6
            )
        )


    ax.set_xlabel(x)
    ax.set_ylabel(y)

    ax.legend(
        handles=legend_handles,
        title=df.index.name or "Group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )

    sns.despine(ax=ax)

def main():
    args = parse_arguments()

    df = load_tsv(args.subject)
    meta = load_tsv(args.meta)

    df = merge_data(df, meta, args.group_col)

    spindle(
        df,
        x=args.x,
        y=args.y,
        figsize=args.figsize,
        ellipses=args.ellipses,
        ellipse_level=args.ellipse_level,
        palette=args.palette,
        group_order=args.group_order
    )

    if args.title_from_subject:
        title = os.path.splitext(os.path.basename(args.subject))[0]
        plt.title(title)

    output = args.output or "spindle.svg"
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved to {output}")

if __name__ == '__main__':
    main()

