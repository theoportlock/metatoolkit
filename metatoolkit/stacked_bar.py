#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_grouped_stacked_bars(
    df: pd.DataFrame,
    bars_col: str,
    color_col: str,
    vals_col: str,
    sig_col: Optional[str],
    sig_threshold: float,
    figsize: tuple,
    horizontal: bool,
    ylabel: str,
    max_categories: int,
    order_bars: bool
):
    """Create a grouped stacked bar plot from long-format data."""

    # Create significance mapping if specified
    sig_map = {}
    if sig_col and sig_col in df.columns:
        for _, row in df.iterrows():
            # Only mark as significant if pval exists and is <= threshold
            if pd.notna(row[sig_col]) and row[sig_col] <= sig_threshold:
                key = (row[bars_col], row[color_col])
                sig_map[key] = True

    # Pivot to get the right structure: bars as index, colors as columns
    pivot_df = df.pivot_table(
        index=bars_col,
        columns=color_col,
        values=vals_col,
        fill_value=0
    )

    # Handle max categories (combine least abundant)
    if len(pivot_df.columns) > max_categories:
        col_means = pivot_df.mean()
        sorted_cols = col_means.sort_values(ascending=False)
        top_cols = sorted_cols.iloc[:max_categories].index
        other_cols = sorted_cols.iloc[max_categories:].index

        pivot_df["Others"] = pivot_df[other_cols].sum(axis=1)
        pivot_df = pivot_df[top_cols.tolist() + ['Others']]

    # Order bars by total value if requested
    if order_bars:
        bar_totals = pivot_df.sum(axis=1).sort_values(ascending=True)
        pivot_df = pivot_df.loc[bar_totals.index]

    # Create the plot
    plot_kind = "barh" if horizontal else "bar"
    ax = pivot_df.plot(
        kind=plot_kind,
        stacked=True,
        figsize=figsize,
        width=0.8,
        cmap="tab20"
    )

    # Set labels based on orientation
    if horizontal:
        ax.set_xlabel(ylabel)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    else:
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add asterisks for significant bars
    if sig_map:
        # Calculate cumulative heights for stacked bars
        cumulative = pivot_df.cumsum(axis=1)

        for bar_idx, bar_name in enumerate(pivot_df.index):
            for col_name in pivot_df.columns:
                if (bar_name, col_name) in sig_map:
                    # Get the position for the asterisk (middle of the bar segment)
                    if col_name == pivot_df.columns[0]:  # First column
                        y_pos = pivot_df.loc[bar_name, col_name] / 2
                    else:
                        prev_cumsum = cumulative.loc[bar_name, pivot_df.columns[pivot_df.columns.get_loc(col_name)-1]]
                        y_pos = prev_cumsum + pivot_df.loc[bar_name, col_name] / 2

                    # Place asterisk
                    if horizontal:
                        ax.text(y_pos, bar_idx, '*', ha='center', va='center',
                               fontweight='bold', fontsize=12, color='white')
                    else:
                        ax.text(bar_idx, y_pos, '*', ha='center', va='center',
                               fontweight='bold', fontsize=12, color='white')

    # Position legend
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return plt

def parse_figsize(figsize_str):
    """Parse figsize string into tuple of floats."""
    try:
        parts = figsize_str.split()
        if len(parts) != 2:
            raise ValueError("Figsize must contain exactly two numbers")
        return (float(parts[0]), float(parts[1]))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid figsize format: {e}")

def main():
    """Create a grouped stacked bar chart from long-format data."""

    parser = argparse.ArgumentParser(
        description="Create grouped stacked bar plots from long-format data."
    )

    # Positional arguments
    parser.add_argument(
        "input",
        type=Path,
        help="TSV file with long-format data"
    )

    # Required arguments
    parser.add_argument(
        "--bars",
        required=True,
        help="Column name for bar groupings (x-axis)"
    )
    parser.add_argument(
        "--color",
        required=True,
        help="Column name for color categories (stacking)"
    )
    parser.add_argument(
        "--vals",
        required=True,
        help="Column name for values"
    )

    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="plot.svg",
        help="Output file path (default: plot.svg)"
    )
    parser.add_argument(
        "--sig",
        help="Column name for significance filtering"
    )
    parser.add_argument(
        "--sig-threshold",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)"
    )
    parser.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(8.0, 6.0),
        help="Figure size as 'width height' (default: 8.0 6.0)"
    )
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="Create horizontal bar plot"
    )
    parser.add_argument(
        "--ylabel",
        default="Value",
        help="Y-axis label (or X-axis for horizontal) (default: Value)"
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Maximum number of color categories before combining others (default: 20)"
    )
    parser.add_argument(
        "--order-bars",
        action="store_true",
        help="Order bars by total value"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"❌ File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Read the data
    try:
        df = pd.read_csv(args.input, sep="\t")
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    required_cols = [args.bars, args.color, args.vals]
    if args.sig:
        required_cols.append(args.sig)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Create the plot
    try:
        plot = plot_grouped_stacked_bars(
            df=df,
            bars_col=args.bars,
            color_col=args.color,
            vals_col=args.vals,
            sig_col=args.sig,
            sig_threshold=args.sig_threshold,
            figsize=args.figsize,
            horizontal=args.horizontal,
            ylabel=args.ylabel,
            max_categories=args.max_categories,
            order_bars=args.order_bars
        )

        plot.tight_layout()
        plot.savefig(args.output, bbox_inches='tight')
        print(f"✅ Saved plot to: {args.output}")

    except Exception as e:
        print(f"❌ Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
