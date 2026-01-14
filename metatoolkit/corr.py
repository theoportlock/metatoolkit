#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path


def fast_spearman(df1, df2=None, fdr=False, min_unique=1, dropna=False):
    """
    Compute pairwise Spearman correlations (within or between datasets), handling NaNs.

    Args:
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame, optional): Second dataset.
        fdr (bool): Apply FDR correction.
        min_unique (int): Minimum unique values per column to retain.
        dropna (bool): Drop rows with any NaNs before correlation.

    Returns:
        pd.DataFrame: Edge list of correlations with p-values and optional q-values.
    """
    df1 = df1.loc[:, df1.nunique() > min_unique].dropna(axis=1, how="all")
    if dropna:
        df1 = df1.dropna(axis=0, how="any")

    if df2 is None:
        # Self-correlation within df1
        valid_cols = df1.columns[df1.count() >= 3]
        df1 = df1[valid_cols]
        if df1.shape[1] < 2:
            return pd.DataFrame()

        cor_matrix = pd.DataFrame(index=df1.columns, columns=df1.columns, dtype=float)
        pval_matrix = pd.DataFrame(index=df1.columns, columns=df1.columns, dtype=float)

        for i, col1 in enumerate(df1.columns):
            for j, col2 in enumerate(df1.columns):
                if i < j:
                    x, y = df1[col1], df1[col2]
                    valid = x.notna() & y.notna()
                    if valid.sum() >= 3:
                        r, p = spearmanr(x[valid], y[valid], nan_policy="omit")
                        cor_matrix.at[col1, col2] = r
                        cor_matrix.at[col2, col1] = r
                        pval_matrix.at[col1, col2] = p
                        pval_matrix.at[col2, col1] = p

    else:
        df2 = df2.loc[:, df2.nunique() > min_unique].dropna(axis=1, how="all")
        if dropna:
            combined = pd.concat([df1, df2], axis=1)
            combined = combined.dropna(axis=0, how="any")
            df1 = combined[df1.columns]
            df2 = combined[df2.columns]

        df1 = df1.loc[:, df1.count() >= 3]
        df2 = df2.loc[:, df2.count() >= 3]
        if df1.empty or df2.empty:
            return pd.DataFrame()

        cor_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=float)
        pval_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=float)

        for col1 in df1.columns:
            for col2 in df2.columns:
                x, y = df1[col1], df2[col2]
                valid = x.notna() & y.notna()
                if valid.sum() >= 3:
                    r, p = spearmanr(x[valid], y[valid], nan_policy="omit")
                    cor_matrix.at[col1, col2] = r
                    pval_matrix.at[col1, col2] = p

    # Convert to edge list
    cor_long = cor_matrix.stack().reset_index()
    cor_long.columns = ["source", "target", "statistic"]

    pval_long = pval_matrix.stack().reset_index()
    pval_long.columns = ["source", "target", "p_value"]

    result = cor_long.merge(pval_long, on=["source", "target"], how="left")

    if fdr and not result.empty:
        result["qval"] = fdrcorrection(result["p_value"].fillna(1))[1]

    return result.dropna(subset=["statistic", "p_value"])


def main():
    parser = argparse.ArgumentParser(description="Compute Spearman correlations efficiently.")
    parser.add_argument("files", nargs="+", help="One or two input files (TSV format with index column).")
    parser.add_argument("-m", "--mult", action="store_true", help="Apply FDR correction (q-values).")
    parser.add_argument("-o", "--output", help="Path for output TSV.")
    parser.add_argument("--dropna", action="store_true", help="Drop rows with any missing values before correlation.")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if len(args.files) == 1:
        df = pd.read_csv(args.files[0], sep="\t", index_col=0)
        output = fast_spearman(df, fdr=args.mult, dropna=args.dropna)
        if not output.empty:
            outfile = args.output or output_dir / f"{Path(args.files[0]).stem}_corr.tsv"
            output.to_csv(outfile, sep="\t", index=False)
        else:
            print("No valid correlations found.")

    elif len(args.files) == 2:
        df1 = pd.read_csv(args.files[0], sep="\t", index_col=0)
        df2 = pd.read_csv(args.files[1], sep="\t", index_col=0)
        output = fast_spearman(df1, df2, fdr=args.mult, dropna=args.dropna)
        if not output.empty:
            outfile = args.output or output_dir / f"{Path(args.files[0]).stem}_{Path(args.files[1]).stem}_corr.tsv"
            output.to_csv(outfile, sep="\t", index=False)
        else:
            print("No valid correlations found.")
    else:
        print("Please provide 1 or 2 files only.")


if __name__ == "__main__":
    main()

