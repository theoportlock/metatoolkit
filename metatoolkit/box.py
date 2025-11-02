#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description='Produces a Boxplot of a given dataset')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument('-x', help='Column name for x-axis')
    parser.add_argument('-y', help='Column name for y-axis')
    parser.add_argument('--hue', help='Column name for hue grouping')
    parser.add_argument(
        '--order',
        help='Comma-separated order of categories for x-axis (or y-axis if horizontal)'
    )
    parser.add_argument(
        '--logy',
        action='store_true',
        help='Set y-axis to log scale (or x-axis if horizontal)'
    )
    parser.add_argument('--show', action='store_true', help='Display the plot window')
    parser.add_argument('--figsize', default='2,2', help='Figure size as width,height')
    parser.add_argument('-o', '--output', help='Output filename without extension')
    parser.add_argument(
        '--meta',
        nargs='+',
        help='Path(s) to metadata file(s) to inner-join with subject data before plotting'
    )
    parser.add_argument(
        '--rc',
        help='Path to matplotlibrc file to use for styling'
    )
    parser.add_argument(
        '--horizontal',
        action='store_true',
        help='Plot horizontally (swap x and y axes)'
    )
    # 🟢 NEW: y-axis (or x-axis if horizontal) limits
    parser.add_argument('--ymin', type=float, help='Minimum value for y-axis')
    parser.add_argument('--ymax', type=float, help='Maximum value for y-axis')
    return parser.parse_args()

def load_data(path_or_name):
    path = Path(path_or_name)
    return pd.read_csv(path, sep='\t', index_col=0)

def merge_meta(df, meta_paths):
    for mpath in meta_paths:
        mdf = pd.read_csv(mpath, sep='\t', index_col=0)
        df = df.join(mdf, how='inner')
    return df

def plot_box(df, x, y, hue, figsize, horizontal=False, order=None):
    df = df.reset_index()
    fig, ax = plt.subplots(figsize=figsize)

    # swap x/y if horizontal
    if horizontal:
        x, y = y, x

    order_list = order.split(",") if order else None

    sns.boxplot(
        data=df,
        x=x or df.columns[0],
        y=y or df.columns[1],
        hue=hue,
        ax=ax,
        order=order_list,
        showfliers=False,
        showcaps=False,
        linewidth=0.4,
        boxprops={'edgecolor': 'black'},
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        capprops={'color': 'black'}
    )
    sns.stripplot(
        data=df,
        x=x or df.columns[0],
        y=y or df.columns[1],
        hue=hue,
        ax=ax,
        order=order_list,
        size=1,
        color='black',
        dodge=bool(hue)
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def save_plots(filename, show):
    filename = Path(filename)
    if filename.suffix:  # If the filename has an extension
        out_path = filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{filename}.svg'

    plt.savefig(out_path)
    if show:
        plt.show()
    plt.clf()

def main():
    args = parse_arguments()

    # load matplotlibrc file if specified
    if args.rc:
        plt.style.use(args.rc)

    # load subject data
    df = load_data(args.subject)

    # optionally merge in metadata explainers
    if args.meta:
        df = merge_meta(df, args.meta)

    print(df.columns)
    # parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    # plot
    ax = plot_box(
        df,
        args.x,
        args.y,
        args.hue,
        figsize,
        horizontal=args.horizontal,
        order=args.order
    )

    # handle log scaling
    if args.logy:
        if args.horizontal:
            ax.set_xscale('log')
        else:
            ax.set_yscale('log')

    # 🟢 NEW: apply y-axis (or x-axis) limits
    if args.horizontal:
        if args.ymin is not None or args.ymax is not None:
            ax.set_xlim(args.ymin, args.ymax)
    else:
        if args.ymin is not None or args.ymax is not None:
            ax.set_ylim(args.ymin, args.ymax)

    plt.tight_layout()

    # save (and optionally show)
    save_plots(args.output or f'{Path(args.subject).stem}_box', args.show)

if __name__ == '__main__':
    main()

