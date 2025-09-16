#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skbio.stats.ordination import cca
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perform Canonical Correspondence Analysis (CCA)")
    parser.add_argument("-d", "--data", required=True,
                       help="Path to response variables data file (TSV)")
    parser.add_argument("-m", "--metadata", required=True,
                       help="Path to metadata file with explanatory variables (TSV)")
    parser.add_argument("-e", "--explanatory_vars", required=True,
                       help="Comma-separated list of explanatory variables (e.g. 'recovery,timepoint,refeed_method')")
    parser.add_argument("-o", "--output", default="results/cca_results",
                       help="Path to output directory")
    parser.add_argument("--scale_data", action="store_true",
                       help="Standardize response variables")
    parser.add_argument("--n_permutations", type=int, default=999,
                       help="Number of permutations for significance testing")

    return parser.parse_args()


def load_and_prepare_data(data_path, metadata_path, explanatory_vars):
    """Load and align the response and metadata tables."""
    # Load data
    response_data = pd.read_csv(data_path, sep="\t", index_col=0)
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    # Align sample IDs
    common_samples = response_data.index.intersection(metadata.index)
    response_data = response_data.loc[common_samples]
    metadata = metadata.loc[common_samples]

    # Select explanatory variables
    X = metadata[explanatory_vars].copy()
    Y = response_data

    return Y, X


def perform_cca(Y, X, scale_data=False):
    """Perform Canonical Correspondence Analysis."""
    if scale_data:
        Y = pd.DataFrame(StandardScaler().fit_transform(Y),
                         index=Y.index, columns=Y.columns)

    # Convert categorical variables to dummies
    X = pd.get_dummies(X, drop_first=True)

    # Run CCA
    result = cca(Y.values, X.values, scaling=2)

    return result, Y, X


def permutation_test(Y, X, observed_eigenvalue, n_permutations=999):
    """Perform permutation test for CCA significance."""
    permuted_eigenvalues = []

    for _ in range(n_permutations):
        permuted_Y = Y.copy().sample(frac=1).values
        permuted_result = cca(permuted_Y, X.values, scaling=2)
        permuted_eigenvalues.append(permuted_result.eigvals[0])

    p_value = (np.sum(np.array(permuted_eigenvalues) >= observed_eigenvalue) + 1) / (n_permutations + 1)

    return p_value


def save_results(result, Y, X, output_dir, p_value=None):
    """Save CCA results to files."""
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame({
        "Eigenvalues": result.eigvals,
        "Explained_variance_percent": result.proportion_explained * 100,
        "Cumulative_variance_percent": np.cumsum(result.proportion_explained * 100)
    })
    summary_df.to_csv(os.path.join(output_dir, "cca_summary.tsv"), sep="\t")

    sample_scores = pd.DataFrame(result.samples, index=Y.index)
    sample_scores.to_csv(os.path.join(output_dir, "sample_scores.tsv"), sep="\t")

    species_scores = pd.DataFrame(result.features, index=Y.columns)
    species_scores.to_csv(os.path.join(output_dir, "species_scores.tsv"), sep="\t")

    biplot_scores = pd.DataFrame(result.biplot_scores, index=X.columns)
    biplot_scores.to_csv(os.path.join(output_dir, "biplot_scores.tsv"), sep="\t")

    if p_value is not None:
        with open(os.path.join(output_dir, "permutation_test.txt"), "w") as f:
            f.write("Permutation test results:\n")
            f.write(f"First eigenvalue: {result.eigvals[0]:.4f}\n")
            f.write(f"P-value: {p_value:.4f}\n")

    return summary_df


def print_summary(summary_df, p_value):
    """Print summary of results to console."""
    print("\nCCA Results Summary:")
    print("="*50)
    print(f"Total variance explained by CCA axes: {summary_df['Explained_variance_percent'].sum():.2f}%")
    print("Variance explained by each axis:")
    for i, row in summary_df.iterrows():
        print(f"  CCA{i+1}: {row['Explained_variance_percent']:.2f}%")
    if p_value:
        print(f"\nPermutation test p-value: {p_value:.4f}")
    print("="*50)


def main():
    args = parse_arguments()

    # Load and prepare data
    Y, X = load_and_prepare_data(args.data, args.metadata, args.explanatory_vars.split(','))

    # Perform CCA
    result, Y, X = perform_cca(Y, X, args.scale_data)

    # Permutation test
    p_value = permutation_test(Y, X, result.eigvals[0], args.n_permutations)

    # Save results
    summary_df = save_results(result, Y, X, args.output, p_value)

    # Print summary
    print_summary(summary_df, p_value)


if __name__ == "__main__":
    main()

