#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ---------- Utility ---------- #

def log(msg):
    print(f"[filter] {msg}")

def load(path):
    """Load a TSV file with index in first column."""
    return pd.read_csv(path, sep="\t", index_col=0)

def save(df, path, index=True):
    """Save DataFrame to a TSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


# ---------- Filtering Logic ---------- #

def apply_filters(df, **kw):
    """Filter DataFrame rows/columns or by one or more columns according to filter_df and other options."""
    df.index = df.index.astype(str)

    # --- Filter by another DataFrame’s index ---
    if (fdf := kw.get("filter_df")) is not None:
        valid_ids = set(fdf.index.astype(str))
        column_arg = kw.get("column", "index")

        columns = [c.strip() for c in column_arg.split(",") if c.strip()]
        keep_mask = pd.Series(True, index=df.index)

        for c in columns:
            if c == "index":
                before = keep_mask.sum()
                keep_mask &= df.index.isin(valid_ids)
                log(f"Index filter: {keep_mask.sum()}/{before} rows remain")
            elif c in df.columns:
                before = keep_mask.sum()
                keep_mask &= df[c].astype(str).isin(valid_ids)
                log(f"Column '{c}' filter: {keep_mask.sum()}/{before} rows remain")
            else:
                log(f"Warning: Column not found: {c}")

        df = df.loc[keep_mask]
        log(f"Filtered by columns {columns}: {len(df)} rows kept")

    # --- Regex filtering ---
    if (rf := kw.get("rowfilt")):
        mask = df.index.str.contains(rf, regex=True)
        df = df.loc[mask]
        log(f"Row regex '{rf}': kept {mask.sum()}/{len(mask)}")

    if (cf := kw.get("colfilt")):
        mask = df.columns.str.contains(cf, regex=True)
        df = df.loc[:, mask]
        log(f"Col regex '{cf}': kept {mask.sum()}/{len(mask)}")

    # --- Value-based filters ---
    if (p := kw.get("prevail")):
        keep = df.astype(bool).sum() > df.shape[0] * p
        df = df.loc[:, keep]
        log(f"Prevalence > {p}: kept {keep.sum()} columns")

    if (a := kw.get("abund")):
        keep = df.mean() > a
        df = df.loc[:, keep]
        log(f"Abundance > {a}: kept {keep.sum()} columns")

    # --- Nonzero filtering ---
    if kw.get("nonzero"):
        before = df.shape
        df = df.loc[df.sum(1) != 0, df.sum() != 0]
        log(f"Removed all-zero rows/cols: {before} → {df.shape}")

    if (m := kw.get("min_nonzero_rows")):
        df = df[df.astype(bool).sum(1) >= m]
        log(f"Min nonzero rows ≥ {m}: kept {df.shape[0]}")

    if (n := kw.get("min_nonzero_cols")):
        df = df.loc[:, df.astype(bool).sum() >= n]
        log(f"Min nonzero cols ≥ {n}: kept {df.shape[1]}")

    # --- Numeric-only filter ---
    if kw.get("numeric_only"):
        df = df.select_dtypes(include=[np.number])
        log(f"Numeric-only columns: {df.shape[1]} remaining")

        # --- Dtype filtering ---
    if (dt := kw.get("dtype")):
        try:
            before = df.shape[1]
            df = df.select_dtypes(include=[dt])
            log(f"Dtype '{dt}' columns: kept {df.shape[1]}/{before}")
        except Exception as e:
            log(f"Dtype filter error: {e}")


    # --- Query filter ---
    if (queries := kw.get("query")):
        qstr = " & ".join(f"({q})" for q in queries)
        before = len(df)
        df = df.query(qstr, engine="python")
        log(f"Query '{qstr}': kept {len(df)}/{before}")

    return None if df.empty else df


# ---------- CLI ---------- #

def parse_args():
    p = argparse.ArgumentParser(description="Filter a TSV file by another TSV’s index or specific column(s).")
    p.add_argument("input", help="Path to input TSV file")
    p.add_argument("-o", "--output", help="Path to output TSV file (default: adds _filter suffix)")
    p.add_argument("-fdf", "--filter_df", help="Path to TSV with index to filter by")
    p.add_argument("-fdfx", "--filter_df_axis", type=int, default=0, help="Axis to filter (0=row, 1=column)")
    p.add_argument("--column", default="index", help="Column(s) to match filter_df index against")
    p.add_argument("-rf", "--rowfilt", help="Regex for filtering rows (index)")
    p.add_argument("-cf", "--colfilt", help="Regex for filtering columns")
    p.add_argument("-p", "--prevail", type=float, help="Prevalence threshold (0–1)")
    p.add_argument("-a", "--abund", type=float, help="Minimum mean abundance threshold")
    p.add_argument("--nonzero", action="store_true", help="Remove all-zero rows and columns")
    p.add_argument("--min_nonzero_rows", type=int, help="Minimum number of non-zero values per row")
    p.add_argument("--min_nonzero_cols", type=int, help="Minimum number of non-zero values per column")
    p.add_argument("--numeric_only", action="store_true", help="Keep numeric columns only")
    p.add_argument("--dtype", help="Keep only columns matching this dtype (e.g. 'float', 'int', 'bool')")
    p.add_argument("-q", "--query", action="append", help="Pandas query string(s)")
    return p.parse_args()


# ---------- Main ---------- #

def main():
    args = parse_args()

    df = load(args.input)
    if args.filter_df:
        args.filter_df = load(args.filter_df)

    out = apply_filters(df, **vars(args))

    if out is None:
        log("All rows and columns filtered out.")
        return

    output_path = args.output or f"{Path(args.input).stem}_filter.tsv"
    save(out, output_path)
    log(f"Saved filtered output to: {output_path}")


if __name__ == "__main__":
    main()

