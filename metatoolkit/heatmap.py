#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path


def load(subject):
    return pd.read_csv(subject, sep='\t', index_col=0)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Produces a faceted heatmap from an edgelist'
    )
    parser.add_argument('subject', help='Path to dataset file')
    parser.add_argument('--sig', default=None, help='Column name for significance values (optional)')
    parser.add_argument('--effect', default='cor', help='Column name for effect size')
    parser.add_argument('--source-col', default='source', help='Column name for y-axis features')
    parser.add_argument('--target-col', default='target', help='Column name for x-axis features')
    parser.add_argument('--sig_thresh', type=float, default=0.05, help='Threshold for drawing significance markers')

    # Faceting controls
    parser.add_argument('--row', help='Column name to facet across rows')
    parser.add_argument('--col', help='Column name to facet across columns')

    # Ordering & Filtering controls
    parser.add_argument('--row-order', help='Comma-separated list specifying row order')
    parser.add_argument('--col-order', help='Comma-separated list specifying column order')
    parser.add_argument(
        '--filter-sig',
        choices=['none', 'source', 'target', 'both'],
        default='none',
        help='Drop rows/cols across all facets if they never reach significance (requires --sig)'
    )

    # Data transformation
    parser.add_argument(
        '--zscore',
        choices=['row', 'col', 'none'],
        default='none',
        help='Z-score the effect matrix by row or column before plotting'
    )
    parser.add_argument(
        '--log1p',
        action='store_true',
        help='Apply sign-preserving log1p transform to effect matrix before plotting'
    )

    # Output controls
    parser.add_argument('-o', '--output', help='Output filepath')
    parser.add_argument(
        '--figsize',
        default='3,3',
        help='Figure size per facet as width,height (default: 3,3)'
    )
    parser.add_argument('--show', action='store_true', help='Display the plot window')

    return parser.parse_args()


def draw_heatmap(data, source_col, target_col, effect_col, sig_col, sig_thresh,
                 global_rows, global_cols, vmin, vmax, **kwargs):
    """Custom mapping function to pivot edgelist and draw heatmap per facet."""
    if data.empty:
        return

    # Pivot long-form data to wide-form matrix
    cor = data.pivot_table(index=source_col, columns=target_col, values=effect_col).fillna(0)

    # Enforce global shape so all subplots align exactly
    cor = cor.reindex(index=global_rows, columns=global_cols).fillna(0)

    ax = plt.gca()

    # Draw the base heatmap (cbar handled globally at the figure level)
    sns.heatmap(
        cor,
        cmap="vlag",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=False,
        yticklabels=True,
        xticklabels=True,
        **kwargs
    )

    # Only process and overlay significance markers if sig_col was provided
    if sig_col and sig_col in data.columns:
        sig = data.pivot_table(index=source_col, columns=target_col, values=sig_col).fillna(1)
        sig = sig.reindex(index=global_rows, columns=global_cols).fillna(1)

        for i, r in enumerate(global_rows):
            for j, c in enumerate(global_cols):
                if sig.loc[r, c] < sig_thresh:
                    ax.text(
                        j + 0.5, i + 0.5, "*",
                        ha="center", va="center", fontsize=10, color='black'
                    )


def main():
    args = parse_arguments()
    df = load(args.subject).reset_index()

    # 1. ESTABLISH GLOBAL ROWS & COLS
    if args.sig and args.sig in df.columns:
        global_sig = df.pivot_table(
            index=args.source_col, columns=args.target_col, values=args.sig, aggfunc='min'
        ).fillna(1)

        mask = global_sig < args.sig_thresh

        if args.filter_sig in ('source', 'both'):
            keep_rows = global_sig.index[mask.any(axis=1)]
        else:
            keep_rows = global_sig.index

        if args.filter_sig in ('target', 'both'):
            keep_cols = global_sig.columns[mask.any(axis=0)]
        else:
            keep_cols = global_sig.columns
    else:
        keep_rows = pd.Index(df[args.source_col].unique())
        keep_cols = pd.Index(df[args.target_col].unique())

    # Apply custom ordering
    if args.row_order:
        row_order_list = args.row_order.split(',')
        global_rows = [r for r in row_order_list if r in keep_rows]
    else:
        global_rows = list(keep_rows)

    if args.col_order:
        col_order_list = args.col_order.split(',')
        global_cols = [c for c in col_order_list if c in keep_cols]
    else:
        global_cols = list(keep_cols)

    # Filter dataframe down to globally kept features
    df = df[df[args.source_col].isin(global_rows) & df[args.target_col].isin(global_cols)].copy()

    # 2. APPLY TRANSFORMATIONS GLOBALLY
    if args.log1p:
        df[args.effect] = np.sign(df[args.effect]) * np.log1p(np.abs(df[args.effect]))

    if args.zscore == 'row':
        df[args.effect] = df.groupby(args.source_col)[args.effect].transform(
            lambda x: (x - x.mean()) / x.std()
        ).fillna(0)
    elif args.zscore == 'col':
        df[args.effect] = df.groupby(args.target_col)[args.effect].transform(
            lambda x: (x - x.mean()) / x.std()
        ).fillna(0)

    # Establish global limits so zero stays white in the 'vlag' colormap
    vmin, vmax = df[args.effect].min(), df[args.effect].max()
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    # 3. SET UP FACET GRID
    width, height = map(float, args.figsize.split(','))
    aspect = width / height if height != 0 else 1

    g = sns.FacetGrid(
        df,
        row=args.row,
        col=args.col,
        height=height,
        aspect=aspect
    )

    # 4. MAP THE HEATMAP TO THE GRID
    g.map_dataframe(
        draw_heatmap,
        source_col=args.source_col,
        target_col=args.target_col,
        effect_col=args.effect,
        sig_col=args.sig,
        sig_thresh=args.sig_thresh,
        global_rows=global_rows,
        global_cols=global_cols,
        vmin=vmin,
        vmax=vmax
    )

    # 5. AESTHETICS & CLEANUP
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # Calculate grid dimensions to handle tick hiding correctly
    n_rows = len(df[args.row].unique()) if args.row else 1
    n_cols = len(df[args.col].unique()) if args.col else 1

    # sns.heatmap forces tick labels onto all subplots. We manually hide inner ones to share axes.
    for i, ax in enumerate(g.axes.flat):
        row_idx = i // n_cols
        col_idx = i % n_cols

        # Y-axis: Only keep labels for the far-left column
        if col_idx == 0:
            ax.tick_params(labelleft=True)
            plt.setp(ax.get_yticklabels(), rotation=0)
            ax.set_ylabel(args.source_col)
        else:
            ax.tick_params(labelleft=False)
            ax.set_ylabel('')

        # X-axis: Only keep labels for the bottom row
        if row_idx == n_rows - 1:
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor')
            ax.set_xlabel(args.target_col)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel('')

    # Add a single, global colorbar on the right
    g.figure.subplots_adjust(right=0.9)
    cbar_ax = g.figure.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    g.figure.colorbar(sm, cax=cbar_ax, label=args.effect)

    # 6. SAVE OUTPUT
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(args.subject).stem}_heatfacets.png"

    plt.savefig(out_path, bbox_inches='tight')
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
