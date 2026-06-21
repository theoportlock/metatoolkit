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
    parser.add_argument(
        '--figsize',
        default='2,2',
        help='Figure size per facet as width,height'
    )
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
    parser.add_argument('--ymin', type=float, help='Minimum value for y-axis')
    parser.add_argument('--ymax', type=float, help='Maximum value for y-axis')

    # Faceting arguments
    parser.add_argument('--row', help='Column name to facet across rows')
    parser.add_argument('--col', help='Column name to facet across columns')
    parser.add_argument(
        '--sharey',
        default='all',
        choices=['all', 'none', 'row', 'col'],
        help='Whether to share the y-axis across facets. Options: all (default), none, row, col'
    )

    # Color palette argument
    parser.add_argument(
        '--palette',
        default='pastel',
        help='Seaborn color palette for the boxplot (default: pastel)'
    )

    return parser.parse_args()

def load_data(path_or_name):
    path = Path(path_or_name)
    return pd.read_csv(path, sep='\t', index_col=0)

def merge_meta(df, meta_paths):
    for mpath in meta_paths:
        mdf = pd.read_csv(mpath, sep='\t', index_col=0)
        df = df.join(mdf, how='inner')
    return df

def plot_box(df, x, y, hue, figsize, horizontal=False, order=None, row=None, col=None, sharey=True, palette='pastel'):
    df = df.reset_index()

    # swap x/y if horizontal
    if horizontal:
        x, y = y, x

    order_list = order.split(",") if order else None

    x_col = x or df.columns[0]
    y_col = y or df.columns[1]

    # NEW: Lock the hue order globally so missing variables in subplots don't shift colors
    if hue:
        hue_order_list = df[hue].dropna().drop_duplicates().tolist()
    else:
        hue_order_list = None

    # Seaborn FacetGrid takes height (per facet) and aspect ratio (width/height)
    width, height = figsize
    aspect = width / height if height != 0 else 1

    g = sns.FacetGrid(
        data=df,
        row=row,
        col=col,
        sharey=sharey,
        height=height,
        aspect=aspect
    )

    g.map_dataframe(
        sns.boxplot,
        x=x_col,
        y=y_col,
        hue=hue,
        order=order_list,
        hue_order=hue_order_list,  # Enforce consistent colors
        palette=palette,
        showfliers=False,
        showcaps=False,
        linewidth=0.4,
        boxprops={'edgecolor': 'black'},
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        capprops={'color': 'black'}
    )

    # Setup Stripplot arguments dynamically to avoid Seaborn Warnings
    strip_kwargs = {
        'x': x_col,
        'y': y_col,
        'order': order_list,
        'size': 1,
        'dodge': bool(hue)
    }

    if hue:
        strip_kwargs['hue'] = hue
        strip_kwargs['hue_order'] = hue_order_list  # Enforce consistent dodge positioning

        # Provide an array of black colors matching the exact number of global hues
        n_hues = len(hue_order_list)
        strip_kwargs['palette'] = ['black'] * n_hues

        # Suppress the stripplot legend so it doesn't overwrite the boxplot colors
        strip_kwargs['legend'] = False
    else:
        strip_kwargs['color'] = 'black'

    g.map_dataframe(
        sns.stripplot,
        **strip_kwargs
    )

    g.despine(right=True, top=True)

    if hue:
        g.add_legend()

    return g

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

    # parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    # Parse sharey argument mapping
    sharey_val = args.sharey
    if sharey_val == 'all':
        sharey_val = True
    elif sharey_val == 'none':
        sharey_val = False

    # plot
    g = plot_box(
        df,
        args.x,
        args.y,
        args.hue,
        figsize,
        horizontal=args.horizontal,
        order=args.order,
        row=args.row,
        col=args.col,
        sharey=sharey_val,
        palette=args.palette
    )

    # handle log scaling across all facet axes
    if args.logy:
        for ax in g.axes.flat:
            if args.horizontal:
                ax.set_xscale('log')
            else:
                ax.set_yscale('log')

    # apply y-axis (or x-axis) limits across the grid
    if args.horizontal:
        if args.ymin is not None or args.ymax is not None:
            g.set(xlim=(args.ymin, args.ymax))
    else:
        if args.ymin is not None or args.ymax is not None:
            g.set(ylim=(args.ymin, args.ymax))

    plt.tight_layout()

    # save (and optionally show)
    save_plots(args.output or f'{Path(args.subject).stem}_box', args.show)

if __name__ == '__main__':
    main()
