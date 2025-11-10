#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_contents, plot as upsetplot_plot


def parse_args():
    p = argparse.ArgumentParser(
        description="Generic UpSet plot: count how many unique IDs are shared across groups."
    )
    p.add_argument("subject", help="Input TSV file (path).")
    p.add_argument("--id", required=True, help="Column name for unique id (e.g. subjectID).")
    p.add_argument("--group", required=True, help="Column name for grouping (e.g. timepoint).")
    p.add_argument("-o", "--output", default="results/upset.svg", help="Output SVG path.")
    p.add_argument("--show-percentages", action="store_true", default=True,
                   help="Show percentages on intersection bars (default: True).")
    p.add_argument("--no-show-percentages", dest="show_percentages", action="store_false",
                   help="Disable percentage labels.")
    p.add_argument("--sort-by-cardinality", action="store_true", default=True,
                   help="Sort intersections by cardinality (largest -> smallest).")
    return p.parse_args()


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    # allow index column in file but not required; read plain TSV
    return pd.read_csv(path, sep="\t", dtype=str)


def build_intersections(df, id_col, group_col):
    """
    Build a dict: group -> set(ids), then convert to upsetplot input via from_contents().
    This ensures that the same id (e.g. subjectID) appearing in multiple groups will be counted as shared.
    """
    # keep only rows where both id and group are present
    df = df[[id_col, group_col]].dropna()
    group_to_ids = df.groupby(group_col)[id_col].apply(set).to_dict()
    return from_contents(group_to_ids)


def plot_upset(intersections, outpath, show_percentages=True, sort_by_cardinality=True):
    """
    Use upsetplot.plot() which supports show_percentages properly.
    sort_by_cardinality True requests the intersections to be ordered by size (largest first).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Build kwargs for upsetplot.plot
    plot_kwargs = {}
    if show_percentages:
        plot_kwargs["show_percentages"] = True

    if sort_by_cardinality:
        plot_kwargs["sort_by"] = "cardinality"

    # Create figure with a reasonable size (user can adjust later)
    fig = plt.figure(figsize=(10, 6))
    # upsetplot_plot accepts the intersections object directly
    upsetplot_plot(intersections, fig=fig, **plot_kwargs)

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UpSet plot to: {outpath}")


def main():
    args = parse_args()
    df = load_data(args.subject)

    # Validate columns
    for col in (args.id, args.group):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input file.")

    intersections = build_intersections(df, args.id, args.group)
    plot_upset(intersections, args.output,
               show_percentages=args.show_percentages,
               sort_by_cardinality=args.sort_by_cardinality)


if __name__ == "__main__":
    main()

