#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import functions as f

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Box - Produces a Boxplot of a given dataset')
    parser.add_argument('subject', help="Path to the input data file")
    parser.add_argument('-x', help="Column for the x-axis")
    parser.add_argument('-y', help="Column for the y-axis")
    parser.add_argument('--hue', help="Column for grouping")
    parser.add_argument('--logy', action='store_true', help="Apply log scale to the y-axis")
    return parser.parse_args()

def boxplot(df, output_path, x=None, y=None, hue=None, logy=False):
    """
    Generate and save a boxplot from the DataFrame.

    Parameters:
    - df: DataFrame to plot
    - output_path: Path to save the plot
    - x: Column for the x-axis
    - y: Column for the y-axis
    - hue: Column for grouping
    - logy: Whether to apply log scale to the y-axis
    """
    # Create boxplot
    sns.boxplot(data=df, x=x, y=y, hue=hue, showfliers=False, linewidth=0.5)
    
    # Add jittered stripplot for individual points
    sns.stripplot(data=df, x=x, y=y, hue=hue, dodge=True if hue else False, 
                  size=1.5, color="0.3", alpha=0.7)
    
    # Apply log scale if requested
    if logy:
        plt.yscale('log')
    
    # Simplify plot appearance
    plt.gca().spines[['right', 'top']].set_visible(False)
    
    # Save plot
    f.savefig(output_path)

def main():
    # Parse arguments
    args = parse_arguments()
    subject = args.subject
    output_path = f"{Path(subject).stem}box"

    # Load data
    df = f.load(subject)

    # Generate plot
    boxplot(df, output_path, x=args.x, y=args.y, hue=args.hue, logy=args.logy)

if __name__ == "__main__":
    main()
