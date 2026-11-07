#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description='Produces a Regplot of a given dataset')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument('-o', '--output', help='Path to output svg')
    parser.add_argument('-x', help='Column name for x-axis')
    parser.add_argument('-y', help='Column name for y-axis')
    parser.add_argument('--hue', help='Column name for hue grouping')
    parser.add_argument('--logy', action='store_true', help='Set y-axis to log scale')
    parser.add_argument('--logx', action='store_true', help='Set x-axis to log scale')
    parser.add_argument('--figsize', help='Figure size as width,height (default: 2,2)', default='2,2')
    parser.add_argument(
        '-m', '--meta',
        nargs='+',
        help='Path(s) to metadata file(s) to inner-join with subject data before plotting'
    )
    return parser.parse_args()

def load_data(subject):
    subject_path = Path(subject)
    if subject_path.is_file():
        return pd.read_csv(subject_path, sep='\t', index_col=0)
    # If not a file, assume subject name and look in results folder
    default_path = Path('results') / f'{subject}.tsv'
    return pd.read_csv(default_path, sep='\t', index_col=0)

def merge_meta(df, meta_paths):
    for mpath in meta_paths:
        mdf = pd.read_csv(mpath, sep='\t', index_col=0)
        df = df.join(mdf, how='inner')
    return df

def plot_reg(df, x=None, y=None, hue=None, ax=None, figsize=(2, 2)):
    # Reset index to ensure the index is part of the data frame
    df = df.reset_index()

    if x is None:
        x = df.columns[0]
    if y is None:
        y = df.columns[1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create the regplot
    #sns.regplot(data=df, x=x, y=y, scatter_kws={"color": "black", 's':2}, line_kws={"color": "red"}, ax=ax)
    sns.regplot(data=df, x=x, y=y, scatter=False, line_kws={"color": "red"}, ax=ax)

    # Overlay scatterplot with points if hue is provided
    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, s=2, legend=False, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, s=2, color='black', ax=ax)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set spine line width to 0.4 (if not handled by rc file)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.4)  # Optional, if not handled by rc file
    ax.spines['bottom'].set_linewidth(0.4)  # Optional, if not handled by rc file

    # Ensure that the ticks reflect the `matplotlibrc` settings
    ax.tick_params(axis='both', which='major', width=0.4, size=4)  # Control tick width and size
    ax.tick_params(axis='both', which='minor', width=0.4, size=2)  # For minor ticks if used

    return ax

def save_plot(filename):
    plt.savefig(filename)
    plt.clf()

def main():
    args = parse_arguments()

    # Parse figsize argument
    figsize = tuple(map(int, args.figsize.split(',')))

    df = load_data(args.subject)

    # Optionally merge in metadata explainers
    if args.meta:
        df = merge_meta(df, args.meta)

    plot_reg(df, x=args.x, y=args.y, hue=args.hue, figsize=figsize)

    if args.logy:
        plt.yscale('log')
    if args.logx:
        plt.xscale('log')

    out_file = args.output if args.output else f'{args.subject}_regplot'
    save_plot(out_file)

if __name__ == '__main__':
    main()
