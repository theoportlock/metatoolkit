#!/usr/bin/env python

import argparse
import pandas as pd
import os
from statsmodels.stats.multitest import multipletests

def load(input_path):
    return pd.read_csv(input_path, sep='\t', index_col=0)

def save(df, output_path):
    outdir = os.path.dirname(output_path)
    if outdir:  # only create if not empty
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=True)

def fdr(df, pcol='pval', method='fdr_bh', alpha=0.05):
    if pcol not in df.columns:
        raise ValueError(
            f"Specified p-value column '{pcol}' not in DataFrame columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # Mask valid (non-NA) p-values
    mask = df[pcol].notna()
    pvals = df.loc[mask, pcol].values

    # Initialise output columns
    df['qval'] = pd.NA
    df['significant'] = False

    # Only apply correction if there are valid p-values
    if len(pvals) > 0:
        rejected, qvals, _, _ = multipletests(
            pvals, alpha=alpha, method=method
        )
        df.loc[mask, 'qval'] = qvals
        df.loc[mask, 'significant'] = rejected

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Apply FDR correction to p-values in a TSV file.")
    parser.add_argument("input", help="Input TSV file with p-values.")
    parser.add_argument("-o", "--outfile", help="Output file path. Default: results/{basename}_fdr.tsv")
    parser.add_argument("-p", "--pcol", default="pval", help="Column name containing p-values. Default: 'pval'")
    parser.add_argument("-m", "--method", default="fdr_bh", help="Correction method (e.g., bonferroni, fdr_bh). Default: fdr_bh")
    parser.add_argument("-a", "--alpha", type=float, default=0.05, help="FDR significance level. Default: 0.05")
    return parser.parse_args()

def main():
    args = parse_args()

    df = load(args.input)
    output = fdr(df, pcol=args.pcol, method=args.method, alpha=args.alpha)

    if args.outfile:
        output_path = args.outfile
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f'results/{base}_fdr.tsv'

    save(output, output_path)
    print(f"Saved FDR-corrected results to: {output_path}")

if __name__ == "__main__":
    main()

