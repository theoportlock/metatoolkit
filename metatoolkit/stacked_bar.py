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
    order_bars: bool,
    log: bool = False,
    max: Optional[float] = None
):
    """Create a grouped stacked bar plot from long-format data."""

    sig_map = {}
    if sig_col and sig_col in df.columns:
        for _, row in df.iterrows():
            if pd.notna(row[sig_col]) and row[sig_col] <= sig_threshold:
                key = (row[bars_col], row[color_col])
                sig_map[key] = True

    pivot_df = df.pivot_table(
        index=bars_col,
        columns=color_col,
        values=vals_col,
        fill_value=0
    )

    if len(pivot_df.columns) > max_categories:
        col_means = pivot_df.mean()
        sorted_cols = col_means.sort_values(ascending=False)
        top_cols = sorted_cols.iloc[:max_categories].index
        other_cols = sorted_cols.iloc[max_categories:].index

        pivot_df["Others"] = pivot_df[other_cols].sum(axis=1)
        pivot_df = pivot_df[top_cols.tolist() + ['Others']]

    if order_bars:
        bar_totals = pivot_df.sum(axis=1).sort_values(ascending=True)
        pivot_df = pivot_df.loc[bar_totals.index]

    plot_kind = "barh" if horizontal else "bar"
    ax = pivot_df.plot(
        kind=plot_kind,
        stacked=True,
        figsize=figsize,
        width=0.8,
        cmap="tab20"
    )

    if horizontal:
        ax.set_xlabel(ylabel)
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    else:
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if log:
        if horizontal:
            ax.set_xscale('log')
        else:
            ax.set_yscale('log')

    if max is not None:
        if horizontal:
            ax.set_xlim(right=max)
        else:
            ax.set_ylim(top=max)

    if sig_map:
        cumulative = pivot_df.cumsum(axis=1)

        for bar_idx, bar_name in enumerate(pivot_df.index):
            for col_name in pivot_df.columns:
                if (bar_name, col_name) in sig_map:
                    if col_name == pivot_df.columns[0]:
                        y_pos = pivot_df.loc[bar_name, col_name] / 2
                    else:
                        prev_cumsum = cumulative.loc[bar_name, pivot_df.columns[pivot_df.columns.get_loc(col_name)-1]]
                        y_pos = prev_cumsum + pivot_df.loc[bar_name, col_name] / 2

                    if horizontal:
                        ax.text(y_pos, bar_idx, '*', ha='center', va='center',
                               fontweight='bold', fontsize=12, color='white')
                    else:
                        ax.text(bar_idx, y_pos, '*', ha='center', va='center',
                               fontweight='bold', fontsize=12, color='white')

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return plt

def main():
    parser = argparse.ArgumentParser(
        description="Create grouped stacked bar plots from long-format data."
    )

    parser.add_argument("input", type=Path, help="TSV file with long-format data")
    parser.add_argument("--bars", required=True, help="Column name for bar groupings (x-axis)")
    parser.add_argument("--color", required=True, help="Column name for color categories (stacking)")
    parser.add_argument("--vals", required=True, help="Column name for values")

    parser.add_argument("-o", "--output", type=Path, default="plot.svg", help="Output file path (default: plot.svg)")
    parser.add_argument("--sig", help="Column name for significance filtering")
    parser.add_argument("--sig-threshold", type=float, default=0.05, help="Significance threshold (default: 0.05)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[8.0, 6.0], help="Figure size as width height (default: 8.0 6.0)")
    parser.add_argument("--horizontal", action="store_true", help="Create horizontal bar plot")
    parser.add_argument("--ylabel", default="Value", help="Y-axis label (or X-axis for horizontal) (default: Value)")
    parser.add_argument("--max-categories", type=int, default=20, help="Maximum number of color categories before combining others (default: 20)")
    parser.add_argument("--order-bars", action="store_true", help="Order bars by total value")
    parser.add_argument("--max", type=float, help="Maximum value for x-axis (or y-axis if vertical)")

    parser.add_argument("--log", action="store_true", help="Apply log scale to x-axis (or y-axis if vertical)")  # <-- added

    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input, sep="\t")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = [args.bars, args.color, args.vals]
    if args.sig:
        required_cols.append(args.sig)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    try:
        plot = plot_grouped_stacked_bars(
            df=df,
            bars_col=args.bars,
            color_col=args.color,
            vals_col=args.vals,
            sig_col=args.sig,
            sig_threshold=args.sig_threshold,
            figsize=tuple(args.figsize),
            horizontal=args.horizontal,
            ylabel=args.ylabel,
            max_categories=args.max_categories,
            order_bars=args.order_bars,
            log=args.log,
            max=args.max
        )

        plot.tight_layout()
        plot.savefig(args.output, bbox_inches='tight')
        print(f"Saved plot to: {args.output}")

    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
