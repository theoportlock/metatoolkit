#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Theo Portlock
"""

import argparse
import pandas as pd
import numpy as np
from itertools import permutations
from scipy.stats import fisher_exact

def fisher_log_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Fisher's exact test for all binary column pairs in a DataFrame.
    Returns a DataFrame with columns: source, target, statistic (log odds), p_value.
    Symmetric entries are included.
    """
    results = []

    for source, target in permutations(df.columns, 2):
        # Build 2x2 contingency table
        contingency = pd.crosstab(df[target], df[source], dropna=False)

        if contingency.shape == (2, 2):
            oddsratio, pvalue = fisher_exact(contingency)
            # Log odds transformation
            if oddsratio == 0:
                log_odds = -np.inf
            elif oddsratio == np.inf:
                log_odds = np.inf
            else:
                log_odds = np.log(oddsratio)
        else:
            log_odds, pvalue = np.nan, np.nan

        results.append({
            "source": source,
            "target": target,
            "statistic": log_odds,
            "p_value": pvalue
        })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Fisher exact test with log odds output.")
    parser.add_argument("file", type=str, help="Path to the input TSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output TSV file.")
    parser.add_argument("--dropna", action="store_true", help="Drop rows with any missing values before testing.")
    args = parser.parse_args()

    df = pd.read_csv(args.file, sep="\t", index_col=0)

    if args.dropna:
        before = df.shape[0]
        df = df.dropna(axis=0, how="any")
        after = df.shape[0]
        print(f"[INFO] Dropped {before - after} rows with missing values ({before} → {after}).")

    out = fisher_log_odds(df)

    if out.empty:
        print("[WARN] No valid binary column pairs found.")
    else:
        print(f"[INFO] Generated {len(out)} Fisher tests.")
        out.to_csv(args.output, sep="\t", index=False)

if __name__ == "__main__":
    main()

