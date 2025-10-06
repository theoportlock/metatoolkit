#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def load(path):
    return pd.read_csv(path, sep="\t", index_col=0)

def save(df, output_path, index=True):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=index)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mann–Whitney U test using Cohen's d + tanh as the statistic."
    )
    parser.add_argument("numeric", type=str, help="Input table with numeric columns")
    parser.add_argument("categorical", type=str, help="Input table with binary categorical columns")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output TSV file path")
    return parser.parse_args()

def cohens_d_tanh(a, b):
    """Compute Cohen's d for two groups and apply tanh to squash between -1 and 1."""
    n1, n2 = len(a), len(b)
    mean1, mean2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)

    # Pooled standard deviation with small epsilon
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2) + 1e-9)

    d = (mean1 - mean2) / s_pooled
    return np.tanh(d)

def mann_whitney_cohensd(df_num, df_cat):
    results = []

    for cat_col in df_cat.columns:
        # Only binary categorical columns
        if df_cat[cat_col].nunique() != 2:
            continue

        for num_col in df_num.columns:
            valid_idx = df_cat[cat_col].notna() & df_num[num_col].notna()
            df_valid = pd.concat([df_num.loc[valid_idx, num_col], df_cat.loc[valid_idx, cat_col]], axis=1)

            groups = [df_valid[df_valid[cat_col] == g][num_col] for g in df_valid[cat_col].unique()]

            if all(len(g) > 0 for g in groups):
                # Mann–Whitney U test
                stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')

                # Cohen's d + tanh statistic
                tstat = cohens_d_tanh(groups[0], groups[1])

                # Original direction
                results.append({
                    "source": cat_col,
                    "target": num_col,
                    "statistic": tstat,
                    "p_value": p,
                    "method": "mwu"
                })
                # Symmetric
                results.append({
                    "source": num_col,
                    "target": cat_col,
                    "statistic": tstat,
                    "p_value": p,
                    "method": "mwu"
                })

    return pd.DataFrame(results)

def main():
    args = parse_args()
    df_num = load(args.numeric)
    df_cat = load(args.categorical)

    output_df = mann_whitney_cohensd(df_num, df_cat)
    print(output_df)

    save(output_df, args.output, index=False)

if __name__ == "__main__":
    main()

