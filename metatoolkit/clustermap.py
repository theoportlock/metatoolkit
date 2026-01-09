#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def load(subject):
    return pd.read_csv(subject, sep='\t', index_col=0)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot - Produces a clustered heatmap from edgelist'
    )
    parser.add_argument('subject')
    parser.add_argument('--sig', default='qval')
    parser.add_argument('--effect', default='cor')
    parser.add_argument('--source-col', default='source')
    parser.add_argument('--target-col', default='target')
    parser.add_argument('--sig_thresh', type=float, default=0.05)

    # Clustering controls
    parser.add_argument('--no-cluster', action='store_true',
                        help='Disable both row and column clustering')
    parser.add_argument('--no-row-cluster', action='store_true')
    parser.add_argument('--no-col-cluster', action='store_true')

    # Ordering controls
    parser.add_argument('--row-order',
                        help='Comma-separated list specifying row order')
    parser.add_argument('--col-order',
                        help='Comma-separated list specifying column order')

    parser.add_argument('--filter-sig',
                        choices=['none', 'source', 'target', 'both'],
                        default='none')
    parser.add_argument('-o', '--output')
    parser.add_argument('--figsize', nargs=2, type=float, default=[4, 4])
    parser.add_argument('--square', action='store_true')

    return parser.parse_args()


def clustermap(df, effect_col, sig_col, source_col, target_col, sig_thresh,
               figsize=(4, 4),
               row_cluster=True, col_cluster=True,
               row_order=None, col_order=None,
               filter_sig='none'):

    df = df.replace([np.inf, -np.inf], np.nan)

    cor = df.reset_index().pivot(
        index=source_col, columns=target_col, values=effect_col
    ).fillna(0)
    sig_df = df.reset_index().pivot(
        index=source_col, columns=target_col, values=sig_col
    ).fillna(1)

    mask = sig_df < sig_thresh

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

    # Apply custom row ordering (if provided)
    if row_order:
        row_list = [r for r in row_order if r in cor.index]
        cor = cor.reindex(row_list)
        sig_df = sig_df.reindex(row_list)

    # Apply custom column ordering (if provided)
    if col_order:
        col_list = [c for c in col_order if c in cor.columns]
        cor = cor.reindex(columns=col_list)
        sig_df = sig_df.reindex(columns=col_list)

    g = sns.clustermap(
        cor,
        cmap="vlag", center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        yticklabels=True,
        xticklabels=True
    )

    # Reindex significance DF to match clustering if clustering is on
    sig_df = sig_df.reindex(
        index=g.data2d.index, columns=g.data2d.columns
    )

    g.ax_heatmap.set_aspect('equal')

    for i, row in enumerate(sig_df.index):
        for j, col in enumerate(sig_df.columns):
            if sig_df.loc[row, col] < sig_thresh:
                g.ax_heatmap.text(
                    j + 0.5, i + 0.5, "*",
                    ha="center", va="center", fontsize=7,
                    transform=g.ax_heatmap.transData
                )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=40, ha='right'
    )

    return g


def reshape_clustermap(cmap, cell_width=0.02, cell_height=0.02):
    ny, nx = cmap.data2d.shape
    w = nx * cell_width
    h = ny * cell_height

    hpos = cmap.ax_heatmap.get_position()

    cmap.ax_heatmap.set_position([hpos.x0, hpos.y0, w, h])

    colpos = cmap.ax_col_dendrogram.get_position()
    cmap.ax_col_dendrogram.set_position([hpos.x0, hpos.y0 + h, w, colpos.height])

    rowpos = cmap.ax_row_dendrogram.get_position()
    cmap.ax_row_dendrogram.set_position([
        hpos.x0 - rowpos.width, hpos.y0, rowpos.width, h
    ])

    if getattr(cmap, 'ax_cbar', None):
        cpos = cmap.ax_cbar.get_position()
        cmap.ax_cbar.set_position([cpos.x0, hpos.y0 + h, cpos.width, cpos.height])


def main():
    args = parse_arguments()
    df = load(args.subject)

    req = {args.source_col, args.target_col, args.effect, args.sig}
    if not req.issubset(df.reset_index().columns):
        raise ValueError(f"Input DataFrame missing required columns: {req}")

    # Determine clustering behavior
    if args.no_cluster:
        row_cluster = False
        col_cluster = False
    else:
        row_cluster = not args.no_row_cluster
        col_cluster = not args.no_col_cluster

    row_order = args.row_order.split(',') if args.row_order else None
    col_order = args.col_order.split(',') if args.col_order else None

    g = clustermap(
        df,
        effect_col=args.effect,
        sig_col=args.sig,
        source_col=args.source_col,
        target_col=args.target_col,
        sig_thresh=args.sig_thresh,
        figsize=tuple(args.figsize),
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_order=row_order,
        col_order=col_order,
        filter_sig=args.filter_sig
    )

    if args.square:
        reshape_clustermap(g)

    output = args.output or f"results/{args.subject}_clustermap.svg"
    plt.savefig(output, bbox_inches='tight')


if __name__ == "__main__":
    main()

