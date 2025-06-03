#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from skbio.stats.ordination import cca
import os
from scipy.stats import f_oneway


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
    # Optionally standardize response variables
    if scale_data:
        Y = pd.DataFrame(StandardScaler().fit_transform(Y), 
                         index=Y.index, columns=Y.columns)

    # Handle categorical variables (convert to dummy variables)
    X = pd.get_dummies(X, drop_first=True)

    # Run CCA
    result = cca(Y.values, X.values, scaling=2)
    
    return result, Y, X


def permutation_test(Y, X, observed_eigenvalue, n_permutations=999):
    """Perform permutation test for CCA significance."""
    permuted_eigenvalues = []
    
    for _ in range(n_permutations):
        permuted_Y = Y.copy().sample(frac=1).values  # Permute rows independently
        permuted_result = cca(permuted_Y, X.values, scaling=2)
        permuted_eigenvalues.append(permuted_result.eigvals[0])  # First eigenvalue

    p_value = (np.sum(np.array(permuted_eigenvalues) >= observed_eigenvalue) + 1) / (n_permutations + 1)
    
    return p_value, permuted_eigenvalues


def save_results(result, Y, X, output_dir, p_value=None):
    """Save all CCA results to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save summary statistics
    summary_df = pd.DataFrame({
        "Eigenvalues": result.eigvals,
        "Explained_variance_percent": result.proportion_explained * 100,
        "Cumulative_variance_percent": np.cumsum(result.proportion_explained * 100)
    })
    summary_df.to_csv(os.path.join(output_dir, "cca_summary.tsv"), sep="\t")

    # Save sample scores
    sample_scores = pd.DataFrame(result.samples, index=Y.index)
    sample_scores.to_csv(os.path.join(output_dir, "sample_scores.tsv"), sep="\t")

    # Save species scores
    species_scores = pd.DataFrame(result.features, index=Y.columns)
    species_scores.to_csv(os.path.join(output_dir, "species_scores.tsv"), sep="\t")

    # Save biplot scores
    biplot_scores = pd.DataFrame(result.biplot_scores, index=X.columns)
    biplot_scores.to_csv(os.path.join(output_dir, "biplot_scores.tsv"), sep="\t")

    # Save permutation test results if available
    if p_value is not None:
        with open(os.path.join(output_dir, "permutation_test.txt"), "w") as f:
            f.write("Permutation test results:\n")
            f.write(f"First eigenvalue: {result.eigvals[0]:.4f}\n")
            f.write(f"P-value: {p_value:.4f}\n")

    return summary_df, sample_scores, species_scores, biplot_scores


def plot_ordination(summary_df, sample_scores, species_scores, biplot_scores, output_dir):
    """Create and save CCA ordination plot."""
    plt.figure(figsize=(10, 8))

    # Plot samples
    sns.scatterplot(x=sample_scores.iloc[:, 0], y=sample_scores.iloc[:, 1], 
                    s=100, alpha=0.7)

    # Plot species (response variables)
    for i, species in enumerate(species_scores.index):
        plt.arrow(0, 0, species_scores.iloc[i, 0], species_scores.iloc[i, 1], 
                  color='r', alpha=0.5, head_width=0.05)
        plt.text(species_scores.iloc[i, 0]*1.15, species_scores.iloc[i, 1]*1.15, 
                 species, color='r', ha='center', va='center')

    # Plot explanatory variables
    for i, var in enumerate(biplot_scores.index):
        plt.arrow(0, 0, biplot_scores.iloc[i, 0], biplot_scores.iloc[i, 1], 
                  color='b', alpha=0.5, head_width=0.05)
        plt.text(biplot_scores.iloc[i, 0]*1.15, biplot_scores.iloc[i, 1]*1.15, 
                 var, color='b', ha='center', va='center')

    plt.xlabel(f"CCA1 ({summary_df.iloc[0, 1]:.1f}%)")
    plt.ylabel(f"CCA2 ({summary_df.iloc[1, 1]:.1f}%)")
    plt.title("CCA Ordination Diagram")
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cca_ordination.png"), dpi=300)
    plt.close()


def print_summary(summary_df, p_value):
    """Print summary of results to console."""
    print("\nCCA Results Summary:")
    print("="*50)
    print(f"Total variance explained by CCA axes: {summary_df['Explained_variance_percent'].sum():.2f}%")
    print(f"Variance explained by each axis:")
    for i, row in summary_df.iterrows():
        print(f"  CCA{i+1}: {row['Explained_variance_percent']:.2f}%")
    if p_value:
        print(f"\nPermutation test p-value: {p_value:.4f}")
    print("="*50)


def main():
    """Main function to execute CCA analysis."""
    args = parse_arguments()
    
    # Load and prepare data
    Y, X = load_and_prepare_data(args.data, args.metadata, args.explanatory_vars.split(','))
    
    # Perform CCA
    result, Y, X = perform_cca(Y, X, args.scale_data)
    
    # Permutation test
    p_value, _ = permutation_test(Y, X, result.eigvals[0], args.n_permutations)
    
    # Save results
    summary_df, sample_scores, species_scores, biplot_scores = save_results(
        result, Y, X, args.output, p_value)
    
    # Create plots
    plot_ordination(summary_df, sample_scores, species_scores, biplot_scores, args.output)
    
    # Print summary
    print_summary(summary_df, p_value)


if __name__ == "__main__":
    main()
