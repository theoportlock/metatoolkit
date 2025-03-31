#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
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
    parser.add_argument('--figsize', help='Figure size as width,height (default: 2,2)', default='2,2')
    return parser.parse_args()

def load_data(subject):
    subject_path = Path(subject)
    if subject_path.is_file():
        return pd.read_csv(subject_path, sep='\t', index_col=0)
    # If not a file, assume subject name and look in ../results folder
    default_path = Path('../results') / f'{subject}.tsv'
    return pd.read_csv(default_path, sep='\t', index_col=0)

def plot_box(df, x=None, y=None, hue=None, ax=None, figsize=(2, 2)):
    # Reset index to ensure the index is part of the data frame
    df = df.reset_index()
    
    if x is None:
        x = df.columns[0]
    if y is None:
        y = df.columns[1]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create the boxplot with all black elements
    sns.boxplot(data=df, x=x, y=y, hue=hue, 
                showfliers=False, 
                showcaps=False, 
                linewidth=0.4, 
                boxprops=dict(edgecolor="black", linewidth=0.4),
                whiskerprops=dict(color="black", linewidth=0.4),
                medianprops=dict(color="black", linewidth=0.4),
                capprops=dict(color="black", linewidth=0.4),
                ax=ax)
    
    # Overlay stripplot with black points
    sns.stripplot(data=df, x=x, y=y, hue=hue, 
                 size=1, 
                 color="black",  # Pure black points
                 linewidth=0.1,  # Thin outline
                 edgecolor="black",  # Black outline
                 ax=ax, 
                 dodge=bool(hue))
    
    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Make remaining spines black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Make ticks black
    ax.tick_params(axis='both', colors='black')
    
    return ax

def save_plots(filename, show=False):
    # Ensure output directory exists
    out_dir = Path('../results')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(filename).stem
    svg_path = out_dir / f'{base_name}.svg'
    pdf_path = out_dir / f'{base_name}.pdf'
    jpg_path = out_dir / f'{base_name}.jpg'
    
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.savefig(jpg_path, dpi=500)
    
    if show:
        plt.show()
    
    # Open the PDF with zathura (if installed)
    subprocess.Popen(['zathura', str(pdf_path)])
    plt.clf()

def main():
    args = parse_arguments()
    
    # Parse figsize argument
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Load data
    df = load_data(args.subject)
    
    # Create boxplot
    ax = plot_box(df, x=args.x, y=args.y, hue=args.hue, figsize=figsize)
    
    if args.logy:
        ax.set_yscale('log')
    
    plt.tight_layout()
    save_plots(f'{Path(args.subject).stem}box', show=args.show)

if __name__ == '__main__':
    main()
