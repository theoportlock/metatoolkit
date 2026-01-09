#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


# -------------------------
# Custom aggregation helpers
# -------------------------

def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


# -------------------------
# CLI
# -------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Group - Groups a dataset and applies aggregation functions"
    )

    parser.add_argument(
        "subject",
        help="Path to input TSV file or subject name (resolved under results/)"
    )

    parser.add_argument(
        "--group_by",
        nargs="+",
        help="Column(s) to group by, or 'all' to aggregate all samples together"
    )

    parser.add_argument(
        "--func",
        nargs="+",
        required=True,
        help="Aggregation function(s): mean, median, sum, std, q25, q75, iqr"
    )

    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        help="Axis for aggregation (default: 0)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Path to save output TSV"
    )

    parser.add_argument(
        "--meta",
        help="Optional metadata TSV to inner-join before grouping"
    )

    return parser.parse_args()


# -------------------------
# I/O helpers
# -------------------------

def load_data(path_or_name):
    path = Path(path_or_name)

    if path.is_file():
        return pd.read_csv(path, sep="\t", index_col=0)

    fallback = Path("results") / f"{path_or_name}.tsv"
    if fallback.is_file():
        return pd.read_csv(fallback, sep="\t", index_col=0)

    raise FileNotFoundError(f"Could not find input file: {path_or_name}")


def merge_meta(df, meta_path, group_by):
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)

    if group_by and not (len(group_by) == 1 and group_by[0].lower() == "all"):
        missing = [c for c in group_by if c not in meta.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

        df = df.join(meta[group_by], how="inner")

    return df


# -------------------------
# Column flattener
# -------------------------

def flatten_columns(midx):
    flat = []
    for entry in midx:
        parts = [str(p) for p in entry if p not in ("", None)]
        flat.append("_".join(parts))
    return flat


# -------------------------
# Core grouping logic
# -------------------------

def group(df, group_by=None, funcs=None, axis=0):
    if funcs is None:
        funcs = ["sum"]

    func_map = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "std": "std",
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
    }

    agg_funcs = []
    for f in funcs:
        if f not in func_map:
            raise ValueError(f"Unsupported aggregation function: {f}")
        agg_funcs.append(func_map[f])

    # ---- Case 1: aggregate all samples ----
    if group_by and len(group_by) == 1 and group_by[0].lower() == "all":
        outdf = df.agg(agg_funcs, axis=axis)

        # outdf: rows = stats, columns = variables
        # Convert to single-row wide format
        outdf = (
            outdf
            .stack()
            .to_frame()
            .T
        )

        outdf.columns = [
            f"{var}_{stat}" for var, stat in outdf.columns
        ]
        outdf.index = ["all"]
        return outdf

    # ---- Case 2: grouped ----
    if group_by:
        outdf = df.groupby(group_by).agg(agg_funcs)
        outdf.columns = flatten_columns(outdf.columns)
        return outdf

    # ---- Case 3: no grouping ----
    outdf = df.agg(agg_funcs, axis=axis)

    if isinstance(outdf, pd.DataFrame):
        outdf = outdf.T
        outdf.columns = flatten_columns(outdf.columns)
    else:
        outdf = pd.DataFrame(outdf).T

    outdf.index = ["all"]
    return outdf


# -------------------------
# Main
# -------------------------

def main():
    args = parse_arguments()

    func_str = "_".join(args.func)
    output = args.output or f"results/{Path(args.subject).stem}_{func_str}.tsv"
    outpath = Path(output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.subject)

    if args.meta:
        df = merge_meta(df, args.meta, args.group_by)

    out = group(
        df,
        group_by=args.group_by,
        funcs=args.func,
        axis=args.axis,
    )

    print(out)
    out.to_csv(outpath, sep="\t")


if __name__ == "__main__":
    main()

