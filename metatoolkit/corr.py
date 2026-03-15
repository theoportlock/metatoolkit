#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
from joblib import Parallel, delayed
from itertools import combinations

def _compute_pair(col1, col2, name1, name2):
    """Worker function to compute a single correlation pair."""
    valid = col1.notna() & col2.notna()
    if valid.sum() >= 3:
        r, p = spearmanr(col1[valid], col2[valid])
        return {"source": name1, "target": name2, "statistic": r, "p_value": p}
    return None

def fast_spearman(df1, df2=None, fdr=False, min_unique=1, dropna=False, n_jobs=-1):
    """
    Compute pairwise Spearman correlations in parallel.
    n_jobs: Number of CPUs to use. -1 means all available.
    """
    # Pre-processing
    df1 = df1.loc[:, df1.nunique() > min_unique].dropna(axis=1, how="all")
    if dropna:
        df1 = df1.dropna(axis=0, how="any")

    if df2 is None:
        valid_cols = df1.columns[df1.count() >= 3]
        df1 = df1[valid_cols]
        if df1.shape[1] < 2:
            return pd.DataFrame()

        # Create unique pairs for internal correlation
        tasks = [
            (df1[c1], df1[c2], c1, c2)
            for c1, c2 in combinations(df1.columns, 2)
        ]
    else:
        df2 = df2.loc[:, df2.nunique() > min_unique].dropna(axis=1, how="all")
        if dropna:
            combined = pd.concat([df1, df2], axis=1)
            combined = combined.dropna(axis=0, how="any")
            df1, df2 = combined[df1.columns], combined[df2.columns]

        df1 = df1.loc[:, df1.count() >= 3]
        df2 = df2.loc[:, df2.count() >= 3]
        if df1.empty or df2.empty:
            return pd.DataFrame()

        # Create pairs for cross-correlation
        tasks = [
            (df1[c1], df2[c2], c1, c2)
            for c1 in df1.columns for c2 in df2.columns
        ]

    # Parallel Execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_pair)(*task) for task in tasks
    )

    # Filter out None results and build DataFrame
    result_df = pd.DataFrame([r for r in results if r is not None])

    if fdr and not result_df.empty:
        # Fill NaN p-values with 1.0 for the correction step
        _, qvals = fdrcorrection(result_df["p_value"].fillna(1))
        result_df["qval"] = qvals

    return result_df

def main():
    parser = argparse.ArgumentParser(description="Compute Spearman correlations in parallel.")
    parser.add_argument("files", nargs="+", help="One or two input TSV files.")
    parser.add_argument("-m", "--mult", action="store_true", help="Apply FDR correction.")
    parser.add_argument("-o", "--output", help="Path for output TSV.")
    parser.add_argument("-p", "--threads", type=int, default=-1, help="Number of threads (-1 for all).")
    parser.add_argument("--dropna", action="store_true", help="Drop rows with any NaNs.")
    args = parser.parse_args()

    if 1 <= len(args.files) <= 2:
        dfs = [pd.read_csv(f, sep="\t", index_col=0) for f in args.files]
        df1 = dfs[0]
        df2 = dfs[1] if len(dfs) == 2 else None

        output = fast_spearman(df1, df2, fdr=args.mult, dropna=args.dropna, n_jobs=args.threads)

        default_name = f"{Path(args.files[0]).stem}_corr.tsv" if df2 is None else \
                       f"{Path(args.files[0]).stem}_{Path(args.files[1]).stem}_corr.tsv"
    else:
        print("Please provide 1 or 2 files only.")
        return

    if output.empty:
        print("No valid correlations found.")
        return

    outfile = Path(args.output) if args.output else Path(default_name)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(outfile, sep="\t", index=False)
    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()
