#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import itertools
import os
import sys
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pairwise Canonical Correlation Analysis (CCA) with missingness logging"
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input TSVs, directories, or glob patterns"
    )
    parser.add_argument(
        "-o", "--output", default="pairwise_cca.tsv",
        help="Path to output summary TSV"
    )
    parser.add_argument(
        "--scale", action="store_true",
        help="Standardize features before CCA"
    )
    parser.add_argument(
        "--n_permutations", type=int, default=999,
        help="Number of permutations for significance testing"
    )
    parser.add_argument(
        "--min_samples", type=int, default=3,
        help="Minimum complete samples required after filtering"
    )
    return parser.parse_args()


# ----------------------------
# Input resolution
# ----------------------------

def resolve_inputs(inputs):
    """
    Expand glob patterns and directories into TSV files.
    """
    files = []

    for inp in inputs:
        # Expand globs
        expanded = glob.glob(inp)
        if expanded:
            for path in expanded:
                if os.path.isdir(path):
                    files.extend(
                        glob.glob(os.path.join(path, "*.tsv"))
                    )
                else:
                    files.append(path)
        else:
            if os.path.isdir(inp):
                files.extend(
                    glob.glob(os.path.join(inp, "*.tsv"))
                )
            else:
                files.append(inp)

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError("No input TSV files resolved from inputs")

    return files


def dataset_name(path):
    """
    Use parent directory + filename stem for uniqueness.
    Example: PCV2/species_short
    """
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{parent}/{stem}"


def load_dataset(path):
    name = dataset_name(path)
    df = pd.read_csv(path, sep="\t", index_col=0)
    return name, df


# ----------------------------
# Missing data handling
# ----------------------------

def drop_missing_pairwise(X, Y, min_samples=3):
    n_total = X.shape[0]

    combined = pd.concat([X, Y], axis=1)
    combined_clean = combined.dropna(axis=0)
    n_complete = combined_clean.shape[0]

    missing_prop = 1.0 - (n_complete / n_total if n_total > 0 else 0.0)

    if n_complete < min_samples:
        return None, None, n_total, n_complete, missing_prop

    return (
        combined_clean[X.columns],
        combined_clean[Y.columns],
        n_total,
        n_complete,
        missing_prop
    )


# ----------------------------
# CCA + permutation test
# ----------------------------

def run_cca(X, Y, scale=False):
    if scale:
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y)
    else:
        X = X.values
        Y = Y.values

    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(X, Y)
    return np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]


def permutation_test(X, Y, observed_corr, n_permutations=999, scale=False):
    perm_corrs = []

    for _ in range(n_permutations):
        Y_perm = Y.sample(frac=1, replace=False).values
        corr = run_cca(
            X,
            pd.DataFrame(Y_perm, index=Y.index),
            scale=scale
        )
        perm_corrs.append(corr)

    perm_corrs = np.array(perm_corrs)
    return (np.sum(perm_corrs >= observed_corr) + 1) / (n_permutations + 1)


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_arguments()

    paths = resolve_inputs(args.inputs)
    datasets = [load_dataset(p) for p in paths]

    results = []

    for (name1, df1), (name2, df2) in itertools.combinations(datasets, 2):
        common = df1.index.intersection(df2.index)

        if len(common) < args.min_samples:
            continue

        X_raw = df1.loc[common]
        Y_raw = df2.loc[common]

        X, Y, n_total, n_complete, missing_prop = drop_missing_pairwise(
            X_raw, Y_raw, min_samples=args.min_samples
        )

        if X is None:
            print(
                f"Skipping {name1} vs {name2}: "
                f"{n_complete}/{n_total} complete "
                f"({missing_prop:.2%} missing)",
                file=sys.stderr
            )
            continue

        corr = run_cca(X, Y, scale=args.scale)
        pval = permutation_test(
            X, Y, corr,
            n_permutations=args.n_permutations,
            scale=args.scale
        )

        results.append([
            name1,
            name2,
            corr,
            pval,
            n_total,
            n_complete,
            missing_prop
        ])

        print(
            f"{name1} vs {name2}: "
            f"corr={corr:.3f}, p={pval:.3f}, "
            f"n={n_complete}/{n_total}, "
            f"missing={missing_prop:.2%}"
        )

    out_df = pd.DataFrame(
        results,
        columns=[
            "source",
            "target",
            "corr",
            "pval",
            "n_samples_total",
            "n_samples_used",
            "missing_proportion"
        ]
    )

    out_df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()

