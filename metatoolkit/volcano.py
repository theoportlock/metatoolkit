#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def volcano(df, hue=None, change='Log2FC', sig='MWW_pval', fc=1, pval=0.05, annot=True, ax=None, size=None):
    if ax is None:
        fig, ax = plt.subplots()

    # Extracting the fold change and p-values
    lfc = df[change]
    pvals = df[sig]
    lpvals = -np.log10(pvals)

    # Set default color for all points
    colors = 'black'
    if hue:
        # Use a red color map
        unique_vals = df[hue].unique()
        color_palette = sns.color_palette("Reds", len(unique_vals))
        color_dict = dict(zip(unique_vals, color_palette))
        colors = df[hue].map(color_dict)

    # Set default size for all points or use the size column
    sizes = 20  # default size
    if size:
        sizes = df[size]

    # Scatter plot
    ax.scatter(lfc, lpvals, c=colors, s=sizes, alpha=0.5)

    # Adding threshold lines
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-np.log10(pval), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fc, color='red', linestyle='-')
    ax.axvline(-fc, color='red', linestyle='-')

    # Setting labels and limits
    ax.set_ylabel('-log10 p-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(lfc).max() * 1.1
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    # Annotate significant points
    sig_species = (lfc.abs() > fc) & (pvals < pval)
    filter_df = df[sig_species]

    if annot:
        for idx, row in filter_df.iterrows():
            ax.text(row[change], -np.log10(row[sig]), s=idx, fontsize=6)

    return ax

parser = argparse.ArgumentParser(description='''
Volcano - Produces a Volcano plot of a given dataset
''')

parser.add_argument('subject', type=str)
parser.add_argument('--change', type=str)
parser.add_argument('--sig', type=str)
parser.add_argument('--fc', type=float)
parser.add_argument('--pval', type=float)
parser.add_argument('--annot', action=argparse.BooleanOptionalAction)

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

# Assemble params
subject = known.get("subject"); known.pop('subject')
change = known.get("change") if known.get("change") else 'coef'
sig = known.get("sig") if known.get("sig") else 'qval'
fc = float(known.get("fc")) if known.get("fc") else 1.0
pval = float(known.get("pval")) if known.get("pval") else 0.05
annot = known.get("annot") if known.get("annot") else True

# Load data
df = f.load(subject)

# Plot
f.setupplot()
volcano(df, change=change, sig=sig, fc=fc, pval=pval, annot=annot)
if os.path.isfile(subject): subject = Path(subject).stem
f.savefig(f'{subject}volcano')
