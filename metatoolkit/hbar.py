#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Horizontal bar plot from coefficients (generalized).')
    parser.add_argument('input', help='Input TSV/CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output path for the SVG file')
    parser.add_argument('--coef-col', required=True, help='Name of the column containing coefficient values')
    parser.add_argument('--hue-col', help='Optional: Name of the column to color bars by (e.g., filename)')
    parser.add_argument('--index-col', help='Optional: Name of the column to use as row labels (otherwise default index)')
    parser.add_argument('--separator', default='\t', help='Field separator (default: tab)')
    parser.add_argument('--xscale', default='linear', choices=['linear', 'log', 'symlog'], help='X-axis scale')
    parser.add_argument('--xlabel', default='Explained variance (R2)', help='Label for the x-axis')
    parser.add_argument('--ylabel', default='Entry', help='Label for the y-axis')
    parser.add_argument('--title', default='Horizontal Barplot', help='Plot title')
    return parser.parse_args()

def plot_horizontal_bar(df, coef_col, hue_col, output_path, xscale, xlabel, ylabel, title):
    # Sort by coefficient column
    df = df.sort_values(coef_col, ascending=False)

    plt.figure(figsize=(5, max(4, len(df) * 0.15)))  # dynamic height

    # Set up barplot parameters
    barplot_kwargs = {
        'x': coef_col,
        'y': df.index,
        'data': df,
        'dodge': False,
        'orient': 'h'
    }

    if hue_col:
        barplot_kwargs['hue'] = hue_col
        barplot_kwargs['palette'] = 'tab10'

    ax = sns.barplot(**barplot_kwargs)

    ax.set_xscale(xscale)
    plt.xlabel(xlabel + (f" ({xscale} scale)" if xscale != 'linear' else ''))
    plt.ylabel(ylabel)
    plt.title(title)

    # Handle legend
    if hue_col:
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # No hue: no legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save plot
    plt.savefig(output_path, format='svg')
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.input, sep=args.separator, index_col=0)

    # Drop NaN values
    df = df.dropna()
    
    # Call plot
    plot_horizontal_bar(
        df,
        coef_col=args.coef_col,
        hue_col=args.hue_col,
        output_path=args.output,
        xscale=args.xscale,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title
    )

if __name__ == '__main__':
    main()

