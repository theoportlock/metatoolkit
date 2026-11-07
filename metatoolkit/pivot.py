#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ---------- Utility ---------- #

def log(msg):
    print(f"[pivot] {msg}")

def load(path, index_col=None):
    """Load a TSV file. Defaults to no index column for long-format data."""
    return pd.read_csv(path, sep="\t", index_col=index_col)

def save(df, path, index=True):
    """Save DataFrame to a TSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


# ---------- Pivoting Logic ---------- #

def apply_pivot(df, **kw):
    """Pivot DataFrame from long to wide format based on CLI options."""

    # Helper to parse comma-separated arguments
    def parse_cols(arg_val):
        if not arg_val:
            return None
        cols = [c.strip() for c in arg_val.split(",") if c.strip()]
        return cols if len(cols) > 1 else cols[0]

    idx_cols = parse_cols(kw.get("index"))
    col_cols = parse_cols(kw.get("columns"))
    val_cols = parse_cols(kw.get("values"))
    agg = kw.get("aggfunc", "mean")
    fill_val = kw.get("fill_value")
    margins = kw.get("margins", False)

    if not idx_cols and not col_cols:
        log("Error: Must specify at least --index or --columns to pivot.")
        return None

    # Cast fill_value to numeric if applicable
    if fill_val is not None:
        try:
            fill_val = float(fill_val)
            if fill_val.is_integer():
                fill_val = int(fill_val)
        except ValueError:
            pass  # Retain as string if it's not a number

    log(f"Pivoting: index={idx_cols}, columns={col_cols}, values={val_cols}, aggfunc={agg}")
    before = df.shape

    try:
        # Create pivot table
        df_pivot = pd.pivot_table(
            df,
            index=idx_cols,
            columns=col_cols,
            values=val_cols,
            aggfunc=agg,
            fill_value=fill_val,
            margins=margins
        )

        # Flatten multi-level columns if requested (common after pivoting multiple values)
        if isinstance(df_pivot.columns, pd.MultiIndex):
            if kw.get("flatten"):
                log("Flattening MultiIndex columns...")
                df_pivot.columns = ["_".join(map(str, col)).strip("_") for col in df_pivot.columns.values]
            else:
                log("Notice: Output contains MultiIndex columns. Use --flatten to merge them into single headers.")

        log(f"Pivot complete: {before} → {df_pivot.shape}")
        return df_pivot

    except Exception as e:
        log(f"Pivot failed: {e}")
        return None


# ---------- CLI ---------- #

def parse_args():
    p = argparse.ArgumentParser(description="Pivot a TSV file from long to wide format.")
    p.add_argument("input", help="Path to input TSV file")
    p.add_argument("-o", "--output", help="Path to output TSV file (default: adds _pivot suffix)")

    p.add_argument("-i", "--index", required=True, help="Column(s) to make the new index (comma-separated)")
    p.add_argument("-c", "--columns", help="Column(s) to make the new columns (comma-separated)")
    p.add_argument("-v", "--values", help="Column(s) to populate the new values (comma-separated)")
    p.add_argument("-a", "--aggfunc", default="mean", help="Aggregation function (e.g., mean, sum, max, min, count, first). Default: mean")

    p.add_argument("-f", "--fill_value", help="Value to replace missing (NA) values in the pivoted table")
    p.add_argument("-m", "--margins", action="store_true", help="Add row/column margins (subtotals / grand totals)")
    p.add_argument("--flatten", action="store_true", help="Flatten hierarchical columns created by multiple values/columns")
    p.add_argument("--index_col", type=int, default=None, help="Column index to use as row labels on load (default: None)")

    return p.parse_args()


# ---------- Main ---------- #

def main():
    args = parse_args()

    df = load(args.input, index_col=args.index_col)

    out = apply_pivot(df, **vars(args))

    if out is None:
        log("No output generated.")
        return

    output_path = args.output or f"{Path(args.input).stem}_pivot.tsv"
    save(out, output_path)
    log(f"Saved pivoted output to: {output_path}")


if __name__ == "__main__":
    main()
