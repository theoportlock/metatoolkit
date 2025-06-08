#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from pathlib import Path
from adjustText import adjust_text

def volcano(df, hue=None, change='Log2FC', sig='MWW_pval', fc=1, pval=0.05,
            annot=True, ax=None, size=None, figsize=(8, 6)):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    lfc = df[change]
    pvals = df[sig]
    lpvals = -np.log10(pvals)

    colors = 'black'
    if hue:
        unique_vals = df[hue].unique()
        color_palette = sns.color_palette("Reds", len(unique_vals))
        color_dict = dict(zip(unique_vals, color_palette))
        colors = df[hue].map(color_dict)

    sizes = 5
    if size:
        sizes = df[size]

    ax.scatter(lfc, lpvals, c=colors, s=sizes, alpha=1)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-np.log10(pval), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fc, color='red', linestyle='-')
    ax.axvline(-fc, color='red', linestyle='-')

    ax.set_ylabel('-log10 p-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(lfc).max() * 1.1
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sig_species = (lfc.abs() > fc) & (pvals < pval)
    filter_df = df[sig_species]

    if annot:
        texts = []
        for idx, row in filter_df.iterrows():
            texts.append(ax.text(row[change], -np.log10(row[sig]), s=idx, fontsize=6))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    return ax


def parse_args():
    parser = argparse.ArgumentParser(description='Volcano - Produces a Volcano plot of a given dataset')
    parser.add_argument('subject', type=str, help='Path to the input TSV file')
    parser.add_argument('--change', type=str, default='coef', help='Column name for log2 fold change')
    parser.add_argument('--sig', type=str, default='qval', help='Column name for significance values (p/q)')
    parser.add_argument('--fc', type=float, default=1.0, help='Fold change threshold')
    parser.add_argument('--pval', type=float, default=0.05, help='P-value threshold')
    parser.add_argument('--annot', action=argparse.BooleanOptionalAction, default=True, help='Annotate significant points')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 6], help='Figure size in inches (width height)')
    parser.add_argument('--output', type=str, help='Output path for the SVG file')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.subject, sep='\t', index_col=0)
    volcano(df, change=args.change, sig=args.sig, fc=args.fc, pval=args.pval,
            annot=args.annot, figsize=tuple(args.figsize))

    if args.output:
        outpath = Path(args.output)
    else:
        outname = Path(args.subject).stem if os.path.isfile(args.subject) else args.subject
        outpath = Path(f'results/{outname}_volcano.svg')

    plt.savefig(outpath)

if __name__ == '__main__':
    main()

