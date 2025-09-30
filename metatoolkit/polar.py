#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def polar_plot(df, output_path="polar_plot.svg", figsize=(4, 4)):
    """
    Creates a polar plot (radar chart) from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Rows represent different categories (e.g., samples),
        Columns represent variables to plot on the polar axes.
    output_path : str
        File path where the polar plot will be saved.
    figsize : tuple
        Figure size (width, height) in inches.
    """

    # Ensure only numeric columns are used
    df = df.select_dtypes(include=[np.number])
    if df.empty:
        raise ValueError("No numeric columns found to plot.")

    # Prepare axis angles for each column (variable)
    categories = df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

    # Close the circle by repeating the first angle
    angles = np.concatenate((angles, [angles[0]]))

    # Generate color palette for each row (sample)
    palette = sns.color_palette("hls", len(df)).as_hex()

    # Create figure and polar axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Make the plot go clockwise and start from the top
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)

    # Plot each row as a separate line
    for idx, (label, row) in enumerate(df.iterrows()):
        values = row.tolist()
        values += [values[0]]  # close the loop
        ax.plot(angles, values, linewidth=2, label=str(label), color=palette[idx])
        ax.fill(angles, values, alpha=0.1, color=palette[idx])  # optional fill

    # Set axis labels and grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.grid(True)

    # Add legend outside the plot
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✔ Polar plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a polar plot (radar chart) from a dataset.")
    parser.add_argument("input", help="Input TSV or CSV file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path for the polar plot (default: same name with _polar.svg).")
    parser.add_argument("--figsize", default="4,4",
                        help="Figure size as 'width,height' in inches (default: 4,4).")

    args = parser.parse_args()

    # Detect delimiter based on file extension
    input_file = args.input
    sep = "\t" if input_file.endswith(".tsv") else ","

    # Load data (first column as index)
    df = pd.read_csv(input_file, sep=sep, index_col=0)

    # Define output path
    output_path = args.output if args.output else os.path.splitext(input_file)[0] + "_polar.svg"

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(",")))

    # Generate and save polar plot
    polar_plot(df, output_path=output_path, figsize=figsize)


if __name__ == "__main__":
    main()

