#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import itertools
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pairwise Canonical Correlation Analysis (CCA)")
    parser.add_argument("inputs", nargs="+",
                        help="List of dataset TSVs (rows=samples, cols=features)")
    parser.add_argument("-o", "--output", default="pairwise_cca.tsv",
                        help="Path to output summary TSV")
    parser.add_argument("--scale", action="store_true",
                        help="Standardize features before CCA")
    parser.add_argument("--n_permutations", type=int, default=999,
                        help="Number of permutations for significance testing")
    return parser.parse_args()


def load_dataset(path):
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, sep="\t", index_col=0)
    return name, df


def run_cca(X, Y, scale=False):
    if scale:
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
    else:
        X, Y = X.values, Y.values

    n_components = min(X.shape[1], Y.shape[1], 1)  # just first canonical dimension
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X, Y)
    corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    return corr


def permutation_test(X, Y, observed_corr, n_permutations=999, scale=False):
    max_corrs = []
    for _ in range(n_permutations):
        Y_perm = Y.sample(frac=1).values  # shuffle rows
        corr = run_cca(X, pd.DataFrame(Y_perm, index=Y.index), scale=scale)
        max_corrs.append(corr)
    p_value = (np.sum(np.array(max_corrs) >= observed_corr) + 1) / (n_permutations + 1)
    return p_value


def main():
    args = parse_arguments()

    # Load all datasets
    datasets = [load_dataset(path) for path in args.inputs]

    results = []
    for (name1, df1), (name2, df2) in itertools.combinations(datasets, 2):
        # Align samples
        common = df1.index.intersection(df2.index)
        if len(common) < 3:
            print(f"Skipping {name1} vs {name2}: not enough overlapping samples")
            continue
        X, Y = df1.loc[common], df2.loc[common]

        corr = run_cca(X, Y, scale=args.scale)
        pval = permutation_test(X, Y, corr, args.n_permutations, scale=args.scale)

        results.append([name1, name2, corr, pval])
        print(f"{name1} vs {name2}: corr={corr:.3f}, p={pval:.3f}")

    # Save results
    out_df = pd.DataFrame(results, columns=["source", "target", "corr", "pval"])
    out_df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()

