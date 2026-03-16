#!/usr/bin/env python

import argparse
import pandas as pd
from pathlib import Path

# -------------------------
# Custom Aggregation Helpers
# -------------------------
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)
def iqr(x): return x.quantile(0.75) - x.quantile(0.25)
def n_nonzero(x): return (x != 0).sum()
def n_total(x): return len(x)
def prev(x): return (x != 0).mean()

AGG_MAP = {
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "std": "std",
    "first": "first",
    "last": "last",
    "q25": q25,
    "q75": q75,
    "iqr": iqr,
    "n_nonzero": n_nonzero,
    "n_total": n_total,
    "prev": prev,
    "count": "count",   # handled specially
}

# -------------------------
# Core grouping logic
# -------------------------
def group(df, group_by=None, funcs=None):
    """
    Group and aggregate a data table.

    - Numeric aggregations operate on numeric columns
    - 'count' returns number of rows per group
    """

    if not group_by:
        raise ValueError("group_by must be provided")

    if not funcs:
        raise ValueError("At least one aggregation function must be provided")

    missing = [f for f in funcs if f not in AGG_MAP]
    if missing:
        raise ValueError(f"Unknown aggregation functions: {missing}")

    want_count = "count" in funcs
    agg_funcs = [AGG_MAP[f] for f in funcs if f != "count"]

    # ---- numeric columns only ----
    value_cols = df.select_dtypes(include="number").columns

    grouped = None

    # ---- numeric aggregations ----
    if agg_funcs:
        if len(value_cols) == 0:
            raise ValueError(
                "Numeric aggregation requested but no numeric columns found"
            )

        grouped = (
            df
            .groupby(group_by)[value_cols]
            .agg(agg_funcs)
        )

        grouped.columns = [
            f"{metric}_{func}"
            for metric, func in grouped.columns
        ]

    # ---- row count aggregation ----
    if want_count:
        counts = df.groupby(group_by).size().to_frame("count")

        if grouped is None:
            grouped = counts
        else:
            grouped = grouped.join(counts)

    return grouped

# -------------------------
# Validation & I/O
# -------------------------
def merge_meta(df, meta_path, group_by):
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)

    if group_by and not (len(group_by) == 1 and group_by[0].lower() == "all"):
        missing = [c for c in group_by if c not in meta.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")
        df = df.join(meta[group_by], how="inner")

    return df

def load_data(path_or_name):
    path = Path(path_or_name)
    if not path.is_file():
        path = Path("results") / f"{path_or_name}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path_or_name}")
    return pd.read_csv(path, sep="\t", index_col=0)

def main():
    parser = argparse.ArgumentParser(description="Group and aggregate datasets.")
    parser.add_argument("subject")
    parser.add_argument("--group_by", nargs="+", required=True)
    parser.add_argument("--func", nargs="+", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--meta")
    args = parser.parse_args()

    df = load_data(args.subject)

    if args.meta:
        df = merge_meta(df, args.meta, args.group_by)

    out = group(df, group_by=args.group_by, funcs=args.func)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t")
    print(out)

if __name__ == "__main__":
    main()

