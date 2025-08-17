#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description='Produces an lmplot of a given dataset with optional metadata and facetting.')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument('-x', help='Column name for x-axis')
    parser.add_argument('-y', help='Column name for y-axis')
    parser.add_argument('--hue', help='Column name for hue grouping')
    parser.add_argument('--col', help='Column name for facet columns')
    parser.add_argument('--row', help='Column name for facet rows')
    parser.add_argument('--logy', action='store_true', help='Set y-axis to log scale')
    parser.add_argument('--logx', action='store_true', help='Set x-axis to log scale')
    parser.add_argument('--show', action='store_true', help='Display the plot window')
    parser.add_argument('--figsize', default='2,2', help='Figure size as width,height')
    parser.add_argument('-o', '--output', help='Output filename without extension')
    parser.add_argument('--meta', nargs='+', help='Metadata file(s) to join with subject data (inner join on index)')
    return parser.parse_args()

def load_data(path_or_name):
    path = Path(path_or_name)
    if path.is_file():
        return pd.read_csv(path, sep='\t', index_col=0)
    else:
        return pd.read_csv(Path('results') / f'{path_or_name}.tsv', sep='\t', index_col=0)

def merge_meta(df, meta_paths):
    for mpath in meta_paths:
        mdf = pd.read_csv(mpath, sep=None, engine='python', index_col=0)
        df = df.join(mdf, how='inner')
    return df

def plot_lm(df, x, y, hue, col, row, figsize, logx=False, logy=False):
    df = df.reset_index()

    if x is None:
        x = df.columns[0]
    if y is None:
        y = df.columns[1]

    height = figsize[1]
    aspect = figsize[0] / figsize[1]

    g = sns.lmplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        height=height,
        aspect=aspect,
        scatter_kws={'s': 2},
        line_kws={'color': 'red'}
    )

    if logx:
        g.set(xscale="log")
    if logy:
        g.set(yscale="log")

    for ax in g.axes.flat:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.tick_params(axis='both', which='major', width=0.4, size=4)
        ax.tick_params(axis='both', which='minor', width=0.4, size=2)

    return g

def save_plot(g, output_file, show=False):
    out_path = Path(output_file)
    if not out_path.suffix:
        out_path = Path('results') / f'{out_path}.svg'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path)
    if show:
        plt.show()
    plt.clf()

def main():
    args = parse_arguments()

    df = load_data(args.subject)

    if args.meta:
        df = merge_meta(df, args.meta)

    figsize = tuple(map(float, args.figsize.split(',')))

    g = plot_lm(
        df, x=args.x, y=args.y, hue=args.hue,
        col=args.col, row=args.row,
        figsize=figsize, logx=args.logx, logy=args.logy
    )

    output = args.output or f'{Path(args.subject).stem}_lmplot'
    save_plot(g, output, show=args.show)

if __name__ == '__main__':
    main()

