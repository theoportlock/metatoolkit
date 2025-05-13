#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject):
        subject = Path(subject).stem
    plt.savefig(f'results/{subject}.svg')
    plt.clf()

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Plot - Produces a clustered heatmap from edgelist')
    parser.add_argument('subject', help="Path to the input data file")
    parser.add_argument('--sig', default='qval', help="Column for significance")
    parser.add_argument('--effect', default='cor', help="Column for effect")
    parser.add_argument('--sig_thresh', type=float, default=0.05, help="Threshold for significance")
    parser.add_argument('--no-cluster', action='store_true', help="Disable clustering")
    parser.add_argument('-o', '--output', help="Path to save the output plot")
    return parser.parse_args()

def clustermap(df, effect_col, sig_col, sig_thresh, figsize=(4,4), cluster=True):
    """
    Generate and display a clustered heatmap with significance markers.
    
    Parameters:
    - df: Input edgelist DataFrame
    - effect_col: Column name for effect size values
    - sig_col: Column name for significance values
    - sig_thresh: Significance threshold
    - figsize: Figure dimensions
    - cluster: Enable clustering
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Pivot to matrices with proper NA handling
    cor = df.reset_index().pivot(index='source', columns='target', values=effect_col)
    sig_df = df.reset_index().pivot(index='source', columns='target', values=sig_col)
    
    # Fill NA values appropriately
    cor = cor.fillna(0)
    sig_df = sig_df.fillna(1)  # Treat missing significance as non-significant

    # Create clustermap with proper clustering control
    g = sns.clustermap(
        cor,
        cmap="vlag",
        center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        row_cluster=cluster,
        col_cluster=cluster,
        yticklabels=True,
        xticklabels=True
    )

    # Align significance data with clustered order
    sig_df = sig_df.reindex(index=g.data2d.index, columns=g.data2d.columns)

    # Add significance markers directly from original values
    for i, row in enumerate(sig_df.index):
        for j, col in enumerate(sig_df.columns):
            try:
                if sig_df.loc[row, col] < sig_thresh:
                    g.ax_heatmap.text(
                        j + 0.5,
                        i + 0.5,
                        "*",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8
                    )
            except KeyError:
                continue  # Skip missing combinations

    # Adjust label rotation
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), 
                                rotation=40, ha='right')
    plt.tight_layout()
    return g

def main():
    args = parse_arguments()
    df = load(args.subject)

    # Verify required columns exist
    if not all(col in df.reset_index().columns for col in ['source', 'target', args.effect, args.sig]):
        raise ValueError("Input dataframe missing required columns")

    g = clustermap(
        df,
        effect_col=args.effect,
        sig_col=args.sig,
        sig_thresh=args.sig_thresh,
        cluster=not args.no_cluster,
    )

    output_path = args.output or f"{args.subject}_clustermap.png"
    savefig(output_path)

if __name__ == "__main__":
    main()
