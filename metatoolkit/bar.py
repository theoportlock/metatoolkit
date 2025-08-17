#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------
# Argument parsing
# ---------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Bar - Produces a barplot of a given dataset'
    )
    parser.add_argument('subject', help='Dataset identifier to load')
    return vars(parser.parse_args())

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject): 
        subject = Path(subject).stem
    plt.savefig(f'results/{subject}.svg')
    plt.clf()

def bar(df, ax=None):
    """
    Create a bar plot with the top 20 columns, summing the rest into "Other".
    
    Parameters:
    - df: Input DataFrame
    - ax: Optional matplotlib Axes object
    """
    maximum = 20  # Hardcoded value
    
    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))

    # Create "Other" category if needed
    if df.shape[1] > maximum:
        top_columns = df.mean().nlargest(maximum - 1).index
        df['Other'] = df.drop(columns=top_columns).sum(axis=1)
        df = df[top_columns.union(['Other'])]

    # Sort and melt data
    df = df[df.median().sort_values(ascending=False).index]
    mdf = df.melt()

    # Create plots using melted columns
    sns.boxplot(
        data=mdf,
        x='variable',
        y='value',
        showfliers=False,
        boxprops=dict(alpha=0.25),
        ax=ax
    )
    
    sns.stripplot(
        data=mdf,
        x='variable',
        y='value',
        size=2,
        color=".3",
        ax=ax
    )

    # Formatting
    ax.set_xlabel('Categories')
    ax.set_ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

# ---------------------------
# Main execution
# ---------------------------

if __name__ == '__main__':
    args = parse_args()
    df = load(args['subject'])
    bar(df)
    plt.tight_layout()
    savefig(f"{args['subject']}_bar")
