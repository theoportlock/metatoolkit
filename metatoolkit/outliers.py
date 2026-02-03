#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- Utility ---------- #

def log(msg):
    print(f"[outlier] {msg}")

def load(path):
    """Load a TSV file with index in first column."""
    return pd.read_csv(path, sep="\t", index_col=0)

def save(df, path, index=True):
    """Save DataFrame to a TSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


# ---------- Outlier Filtering ---------- #

def filter_outliers(df, iqr_k=1.5, z_thresh=None, columns=None, drop_na=False):
    """
    Mask outliers in numeric columns using IQR or Z-score filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    iqr_k : float
        Multiplicative factor for IQR filtering
    z_thresh : float or None
        Threshold for Z-score filtering
    columns : list[str] or None
        List of columns to filter; default all numeric columns
    drop_na : bool
        Drop rows/columns that become all-NaN after masking
    """
    df.index = df.index.astype(str)
    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        log("No numeric columns found, skipping outlier filtering")
        return df

    if columns:
        columns = [c for c in columns if c in numeric.columns]
    else:
        columns = numeric.columns.tolist()

    if not columns:
        log("No valid numeric columns selected, skipping outlier filtering")
        return df

    # --- IQR-based filtering (default) ---
    if iqr_k is not None:
        Q1 = numeric[columns].quantile(0.25)
        Q3 = numeric[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_k * IQR
        upper = Q3 + iqr_k * IQR

        mask = ~((numeric[columns] < lower) | (numeric[columns] > upper))
        df[columns] = numeric[columns].where(mask)

        # Logging per column
        for col in columns:
            total = len(df)
            n_outliers = (~mask[col]).sum()
            pct = n_outliers / total * 100
            log(f"IQR filter (k={iqr_k}) '{col}': {n_outliers}/{total} ({pct:.1f}%) outliers masked")

    # --- Z-score filtering ---
    if z_thresh is not None:
        mu = numeric[columns].mean()
        sigma = numeric[columns].std(ddof=0)
        zscores = (numeric[columns] - mu) / sigma
        mask = zscores.abs() <= z_thresh
        df[columns] = numeric[columns].where(mask)

        for col in columns:
            total = len(df)
            n_outliers = (~mask[col]).sum()
            pct = n_outliers / total * 100
            log(f"Z-score filter (|z|>{z_thresh}) '{col}': {n_outliers}/{total} ({pct:.1f}%) outliers masked")

    # --- Drop all-NaN rows/columns after masking ---
    if drop_na:
        before = df.shape
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        log(f"Dropped all-NaN rows/cols after masking: {before} → {df.shape}")

    return df


# ---------- CLI ---------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Mask or remove numeric outliers in a TSV file using IQR (default) or Z-score methods."
    )
    p.add_argument("input", help="Path to input TSV file")
    p.add_argument("-o", "--output", help="Path to output TSV file (default: adds _outliers suffix)")
    p.add_argument("--columns", help="Comma-separated list of numeric columns to filter (default: all)")
    p.add_argument("--outlier_iqr", type=float, default=1.5, help="Mask outliers using IQR*k (default: 1.5)")
    p.add_argument("--outlier_zscore", type=float, help="Mask outliers with |z| > threshold")
    p.add_argument("--drop_outlier_na", action="store_true", help="Drop rows/cols that become all-NaN after masking")
    return p.parse_args()


# ---------- Main ---------- #

def main():
    args = parse_args()
    df = load(args.input)

    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None

    out = filter_outliers(
        df,
        iqr_k=args.outlier_iqr,
        z_thresh=args.outlier_zscore,
        columns=columns,
        drop_na=args.drop_outlier_na
    )

    if out.empty:
        log("All rows and columns removed after outlier filtering.")
        return

    output_path = args.output or f"{Path(args.input).stem}_outliers.tsv"
    save(out, output_path)
    log(f"Saved filtered output to: {output_path}")


if __name__ == "__main__":
    main()

