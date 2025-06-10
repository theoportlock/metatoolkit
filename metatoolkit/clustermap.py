#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot - Produces a clustered heatmap from edgelist'
    )
    parser.add_argument('subject', help="Path to the input data file")
    parser.add_argument('--sig', default='qval', help="Column for significance")
    parser.add_argument('--effect', default='cor', help="Column for effect")
    parser.add_argument(
        '--sig_thresh', type=float, default=0.05,
        help="Threshold for significance"
    )
    parser.add_argument(
        '--no-cluster', action='store_true',
        help="Disable clustering"
    )
    parser.add_argument('-o', '--output', help="Path to save the output plot")
    parser.add_argument(
        '--filter-sig',
        choices=['none', 'source', 'target', 'both'],
        default='none',
        help="Filter to include only source, target, or both with ≥1 significant result"
    )
    parser.add_argument(
        '--figsize', nargs=2, type=float, default=[4, 4],
        metavar=('WIDTH', 'HEIGHT'),
        help="Figure size in inches (width height), e.g. --figsize 6 6"
    )
    parser.add_argument(
        '--square', action='store_true',
        help="Resize axes so each heatmap cell is square"
    )
    return parser.parse_args()

def clustermap(df, effect_col, sig_col, sig_thresh,
               figsize=(4,4), cluster=True, filter_sig='none'):
    """Generate and return a clustered heatmap with centered significance markers."""
    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Pivot to matrices
    cor = df.reset_index().pivot(
        index='source', columns='target', values=effect_col
    ).fillna(0)
    sig_df = df.reset_index().pivot(
        index='source', columns='target', values=sig_col
    ).fillna(1)

    mask = sig_df < sig_thresh

    # Apply filtering
    if filter_sig in ('source', 'both'):
        keep_rows = mask.any(axis=1)
    else:
        keep_rows = slice(None)
    if filter_sig in ('target', 'both'):
        keep_cols = mask.any(axis=0)
    else:
        keep_cols = slice(None)

    cor = cor.loc[keep_rows, keep_cols]
    sig_df = sig_df.loc[keep_rows, keep_cols]

    # Draw clustermap
    g = sns.clustermap(
        cor,
        cmap="vlag", center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        row_cluster=cluster,
        col_cluster=cluster,
        yticklabels=True,
        xticklabels=True
    )

    # Reorder significance DataFrame to match the clustering
    sig_df = sig_df.reindex(
        index=g.data2d.index, columns=g.data2d.columns
    )

    # 1) True square cells
    g.ax_heatmap.set_aspect('equal')

    # 2) Centered asterisks
    for i, row in enumerate(sig_df.index):
        for j, col in enumerate(sig_df.columns):
            if sig_df.loc[row, col] < sig_thresh:
                g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    "*",
                    ha="center", va="center",
                    fontsize=7,
                    transform=g.ax_heatmap.transData
                )

    # Tidy labels
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=40, ha='right'
    )

    return g

def reshape_clustermap(cmap, cell_width=0.02, cell_height=0.02):
    """Re-anchor dendrograms & cbar after manual heatmap resizing."""
    # Compute new heatmap size
    ny, nx = cmap.data2d.shape
    w = nx * cell_width
    h = ny * cell_height

    # Original heatmap position
    hpos = cmap.ax_heatmap.get_position()

    # 1) Heatmap
    cmap.ax_heatmap.set_position([hpos.x0, hpos.y0, w, h])

    # 2) Column dendrogram above
    colpos = cmap.ax_col_dendrogram.get_position()
    cmap.ax_col_dendrogram.set_position([
        hpos.x0,
        hpos.y0 + h,
        w,
        colpos.height
    ])

    # 3) Row dendrogram to left
    rowpos = cmap.ax_row_dendrogram.get_position()
    new_x0 = hpos.x0 - rowpos.width
    cmap.ax_row_dendrogram.set_position([
        new_x0,
        hpos.y0,
        rowpos.width,
        h
    ])

    # 4) Colorbar (if present) atop the heatmap
    if getattr(cmap, 'ax_cbar', None):
        cpos = cmap.ax_cbar.get_position()
        cmap.ax_cbar.set_position([
            cpos.x0,
            hpos.y0 + h,
            cpos.width,
            cpos.height
        ])


def main():
    args = parse_arguments()
    df = load(args.subject)

    # Validate
    req = {'source', 'target', args.effect, args.sig}
    if not req.issubset(df.reset_index().columns):
        raise ValueError("Input dataframe missing required columns")

    g = clustermap(
        df,
        effect_col=args.effect,
        sig_col=args.sig,
        sig_thresh=args.sig_thresh,
        cluster=not args.no_cluster,
        figsize=tuple(args.figsize),
        filter_sig=args.filter_sig
    )

    if args.square:
        reshape_clustermap(g)

    output = args.output or f"results/{args.subject}_clustermap.svg"
    plt.savefig(output, bbox_inches='tight')


if __name__ == "__main__":
    main()

