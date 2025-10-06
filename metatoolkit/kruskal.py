#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from scipy.stats import kruskal
from itertools import combinations

def load(path):
    return pd.read_csv(path, sep="\t", index_col=0)

def save(df, output_path, index=True):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=index)

def parse_args():
    parser = argparse.ArgumentParser(description="Kruskal-Wallis test: numeric vs categorical columns.")
    parser.add_argument("numeric", type=str, help="Input table containing numeric columns")
    parser.add_argument("categorical", type=str, help="Input table containing categorical columns")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output TSV file path")
    return parser.parse_args()

def kruskal_wallis_test(df_num, df_cat):
    # Merge into one DataFrame so any column can be compared to any other
    df_all = pd.concat([df_num, df_cat], axis=1)

    results = []
    cols = df_all.columns

    for colA, colB in combinations(cols, 2):
        entry = {
            "source": colA,
            "target": colB,
            "statistic": float("nan"),
            "p_value": float("nan")
        }

        # Compute Kruskal-Wallis in both directions if possible
        try:
            # Treat colA as groups, colB as values
            valid_idx = df_all[colA].notna() & df_all[colB].notna()
            groups = [df_all[colB][valid_idx & (df_all[colA] == g)] for g in df_all[colA].unique()]

            if len(groups) > 1 and all(len(g) > 0 for g in groups) and not all(g.nunique() == 1 for g in groups):
                stat, p = kruskal(*groups)
                entry["statistic"] = stat
                entry["p_value"] = p

        except Exception:
            pass

        # Save (A,B)
        results.append(entry)

        # Mirror (B,A) with identical values
        mirrored = {
            "source": colB,
            "target": colA,
            "statistic": entry["statistic"],
            "p_value": entry["p_value"]
        }
        results.append(mirrored)

    return pd.DataFrame(results)


def main():
    args = parse_args()

    df_num = load(args.numeric)
    df_cat = load(args.categorical)

    output_df = kruskal_wallis_test(df_num, df_cat)
    print(output_df)

    save(output_df, args.output, index=False)

if __name__ == "__main__":
    main()

