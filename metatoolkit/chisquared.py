#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from itertools import permutations
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association


def load(path):
    return pd.read_csv(path, sep="\t", index_col=0)


def save(df, output_path, index=True):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Chi-Squared test and Cramér's V for all pairs of categorical columns in a TSV file."
    )
    parser.add_argument("input", type=str, help="Input TSV file containing categorical variables")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output TSV file path")
    return parser.parse_args()


def chi_squared(df: pd.DataFrame) -> pd.DataFrame:
    column_pairs = list(permutations(df.columns, 2))
    results = {
        "statistic": [],  # Cramér's V
        "p_value": [],    # From chi2
    }

    for source, target in column_pairs:
        contingency_table = pd.crosstab(df[target], df[source])
        chi2_stat, pvalue, dof, _ = chi2_contingency(contingency_table)
        v = association(contingency_table, method="cramer")

        results["statistic"].append(v)
        results["p_value"].append(pvalue)

    index = pd.MultiIndex.from_tuples(column_pairs, names=["source", "target"])
    result_df = pd.DataFrame(results, index=index)
    return result_df


def main():
    args = parse_args()
    df = load(args.input)

    output_df = chi_squared(df)
    print(output_df)

    save(output_df, args.output)


if __name__ == "__main__":
    main()

