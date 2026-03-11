#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import os
from statsmodels.stats.multitest import multipletests

def load(input_path):
    # Changed index_col to None by default to avoid losing data
    # Standard TSVs often don't have a formal index
    return pd.read_csv(input_path, sep="\t")

def save(df, output_path):
    outdir = os.path.dirname(output_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False) # index=False is usually cleaner for TSVs

def apply_fdr(df, pcol="pval", method="fdr_bh", alpha=0.05, colname="fdr", replace=False):
    if pcol not in df.columns:
        raise ValueError(
            f"Specified p-value column '{pcol}' not in DataFrame columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # Ensure p-values are numeric, coercing errors to NaN
    df[pcol] = pd.to_numeric(df[pcol], errors='coerce')

    # Mask to ignore NaNs during calculation
    mask = df[pcol].notna()
    pvals = df.loc[mask, pcol].values

    # Initialize results column with np.nan (float type)
    corrected_values = np.full(len(df), np.nan)

    if len(pvals) > 0:
        # reject is a boolean array, corrected is the array of q-values
        reject, corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
        corrected_values[mask] = corrected

    target_col = pcol if replace else colname
    df[target_col] = corrected_values

    return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply FDR correction (Benjamini-Hochberg) to p-values in a TSV file."
    )
    parser.add_argument("input", help="Input TSV file.")
    parser.add_argument("-o", "--outfile", help="Output file path.")
    parser.add_argument("-p", "--pcol", default="pval", help="Column with p-values (default: 'pval').")
    parser.add_argument("-m", "--method", default="fdr_bh", help="Correction method (e.g., bonferroni, fdr_bh).")
    parser.add_argument("-a", "--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("-c", "--colname", default="fdr", help="New column name (default: 'fdr').")
    parser.add_argument("--replace", action="store_true", help="Overwrite the original p-value column.")

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        return

    df = load(args.input)

    output = apply_fdr(
        df,
        pcol=args.pcol,
        method=args.method,
        alpha=args.alpha,
        colname=args.colname,
        replace=args.replace,
    )

    output_path = args.outfile

    save(output, output_path)
    print(f"Successfully processed {len(df)} rows.")
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    main()
