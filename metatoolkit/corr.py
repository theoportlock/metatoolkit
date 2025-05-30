#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

def fast_spearman(df1, df2=None, fdr=False, min_unique=1):
    """
    Compute Spearman correlations efficiently, handling NaNs.

    Args:
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame, optional): Second dataset for cross-correlations.
        fdr (bool): Apply FDR correction.
        min_unique (int): Minimum unique values per column.

    Returns:
        pd.DataFrame: Correlation results in edge list format.
    """
    # Drop columns that have fewer than min_unique unique values
    df1 = df1.loc[:, df1.nunique() > min_unique].dropna(axis=1, how="all")

    if df2 is None:
        # Remove columns with fewer than 3 non-NaN values
        valid_cols = df1.columns[df1.count() >= 3]
        df1 = df1[valid_cols]

        if df1.shape[1] < 2:
            return pd.DataFrame()

        cor_matrix = df1.corr(method='spearman', min_periods=3)

        # Compute p-values manually
        pval_matrix = pd.DataFrame(np.nan, index=df1.columns, columns=df1.columns)
        for i, col1 in enumerate(df1.columns):
            for j, col2 in enumerate(df1.columns):
                if i < j:
                    x, y = df1[col1], df1[col2]
                    valid = x.notna() & y.notna()
                    if valid.sum() >= 3:
                        _, p = spearmanr(x[valid], y[valid], nan_policy='omit')
                        pval_matrix.at[col1, col2] = p
                        pval_matrix.at[col2, col1] = p  # Symmetric

    else:
        df2 = df2.loc[:, df2.nunique() > min_unique].dropna(axis=1, how="all")

        # Remove columns with fewer than 3 valid values
        df1 = df1.loc[:, df1.count() >= 3]
        df2 = df2.loc[:, df2.count() >= 3]

        if df1.shape[1] == 0 or df2.shape[1] == 0:
            return pd.DataFrame()

        cor_matrix = df1.corrwith(df2, method='spearman', min_periods=3)

        # Compute p-values manually
        pval_matrix = pd.DataFrame(np.nan, index=df1.columns, columns=df2.columns)
        for col1 in df1.columns:
            for col2 in df2.columns:
                x, y = df1[col1], df2[col2]
                valid = x.notna() & y.notna()
                if valid.sum() >= 3:
                    _, p = spearmanr(x[valid], y[valid], nan_policy='omit')
                    pval_matrix.at[col1, col2] = p

    # Convert matrices to edge list format
    cor_long = cor_matrix.stack().reset_index()
    cor_long.columns = ["source", "target", "cor"]

    pval_long = pval_matrix.stack().reset_index()
    pval_long.columns = ["source", "target", "pval"]

    result = cor_long.merge(pval_long, on=["source", "target"], how="left")

    # Apply FDR correction if requested
    if fdr and not result.empty:
        result["qval"] = fdrcorrection(result["pval"].fillna(1))[1]

    return result.dropna()

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Compute Spearman correlations efficiently.")
parser.add_argument("files", nargs="+", help="Input file(s)")
parser.add_argument("-m", "--mult", action="store_true", help="Apply FDR correction")
args = parser.parse_args()

# Load and process data
if len(args.files) == 1:
    df = pd.read_csv(args.files[0], sep='\t', index_col=0)
    output = fast_spearman(df, fdr=args.mult)
    if not output.empty:
        output.to_csv('results/' + Path(args.files[0]).stem + ".corr.tsv", sep='\t', index=False)
    else:
        print("No valid correlations found.")

elif len(args.files) == 2:
    df1 = pd.read_csv(args.files[0], sep='\t', index_col=0)
    df2 = pd.read_csv(args.files[1], sep='\t', index_col=0)
    output = fast_spearman(df1, df2, fdr=args.mult)
    if not output.empty:
        output.to_csv('results/' + Path(args.files[0]).stem + "_" + args.files[1] + ".corr.tsv", sep='\t', index=False)
    else:
        print("No valid correlations found.")

else:
    print("Please provide 1 or 2 files.")
