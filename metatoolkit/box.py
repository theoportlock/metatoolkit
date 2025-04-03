#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import subprocess
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description='Produces a Boxplot of a given dataset')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument('-x', help='Column name for x-axis')
    parser.add_argument('-y', help='Column name for y-axis')
    parser.add_argument('--hue', help='Column name for hue grouping')
    parser.add_argument('--logy', action='store_true', help='Set y-axis to log scale')
    parser.add_argument('--show', action='store_true', help='Display the plot window')
    parser.add_argument('--figsize', default='2,2', help='Figure size as width,height')
    parser.add_argument('-o', '--output', help='Output filename without extension')
    return parser.parse_args()

def load_data(subject):
    path = Path(subject) if Path(subject).is_file() else Path('../results') / f'{subject}.tsv'
    return pd.read_csv(path, sep='\t', index_col=0)

def plot_box(df, x, y, hue, figsize):
    df = df.reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x or df.columns[0], y=y or df.columns[1], hue=hue, ax=ax,
                showfliers=False, showcaps=False, linewidth=0.4,
                boxprops={'edgecolor': 'black'}, whiskerprops={'color': 'black'},
                medianprops={'color': 'black'}, capprops={'color': 'black'})
    sns.stripplot(data=df, x=x, y=y, hue=hue, ax=ax, size=1, color='black', dodge=bool(hue))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def save_plots(filename, show):
    out_dir = Path('../results')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'{filename}.svg')
    plt.clf()

def main():
    args = parse_arguments()
    df = load_data(args.subject)
    figsize = tuple(map(float, args.figsize.split(',')))
    ax = plot_box(df, args.x, args.y, args.hue, figsize)
    if args.logy:
        ax.set_yscale('log')
    plt.tight_layout()
    save_plots(args.output or f'{Path(args.subject).stem}box', args.show)

if __name__ == '__main__':
    main()
