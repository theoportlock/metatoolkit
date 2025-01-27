#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import skbio
import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate and plot a PCoA biplot from metagenomic species profiles.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Metagenomic species profile file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path for the plot image")
    return parser.parse_args()

def calculate_distance_matrix(profile_df):
    """
    Calculate a Bray-Curtis distance matrix from metagenomic species profiles.

    Parameters:
    profile_df (pd.DataFrame): DataFrame where rows are samples and columns are species or features.

    Returns:
    skbio.DistanceMatrix: A Bray-Curtis distance matrix.
    """
    # Convert to a presence/absence or normalized abundance matrix if necessary
    distance_matrix = skbio.diversity.beta_diversity("braycurtis", profile_df.values, ids=profile_df.index)
    return distance_matrix

def perform_pcoa(distance_matrix):
    """
    Perform PCoA on the given distance matrix.

    Parameters:
    distance_matrix (skbio.DistanceMatrix): A distance matrix for PCoA.

    Returns:
    tuple: (pcoa_results, explained_variance) where `pcoa_results` is a DataFrame with
           PCoA coordinates and `explained_variance` is a Series with variance ratios.
    """
    PCoA = skbio.stats.ordination.pcoa(distance_matrix, number_of_dimensions=2)
    explained_variance = PCoA.proportion_explained
    return PCoA.samples, explained_variance

def plot_pcoa_biplot(pcoa_df, explained_variance, feature_loadings, output=None):
    """
    Plot a PCoA biplot with samples and feature loadings.

    Parameters:
    pcoa_df (pd.DataFrame): DataFrame containing the PCoA results for each sample.
    explained_variance (pd.Series): Series containing the variance explained by each PCoA axis.
    feature_loadings (pd.DataFrame): DataFrame with loadings of each feature on the PCoA axes.
    output (str): Path to save the plot image. If None, shows the plot interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pcoa_df.iloc[:, 0], pcoa_df.iloc[:, 1], color='blue', label='Samples')

    # Plot feature loadings as arrows
    for i, feature in enumerate(feature_loadings.index):
        ax.arrow(0, 0, feature_loadings.iloc[i, 0] * 0.1, feature_loadings.iloc[i, 1] * 0.1, 
                 color='red', alpha=0.5)
        ax.text(feature_loadings.iloc[i, 0] * 0.12, feature_loadings.iloc[i, 1] * 0.12, feature,
                color='red', ha='center', va='center')

    ax.set_xlabel(f'PCo1 ({explained_variance[0]:.1%})')
    ax.set_ylabel(f'PCo2 ({explained_variance[1]:.1%})')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.legend()
    ax.set_title('PCoA Biplot with Sample Scores and Feature Loadings')

    if output:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

def calculate_feature_loadings(profile_df, pcoa_df):
    """
    Calculate feature loadings by correlating each feature with the PCoA axes.

    Parameters:
    profile_df (pd.DataFrame): Original species profiles DataFrame.
    pcoa_df (pd.DataFrame): DataFrame containing the PCoA results.

    Returns:
    pd.DataFrame: DataFrame with loadings of each feature on the first two PCoA axes.
    """
    loadings = pd.DataFrame(index=profile_df.columns)
    for axis in range(2):  # Only consider the first two axes
        loadings[f"PCo{axis + 1}"] = profile_df.corrwith(pcoa_df.iloc[:, axis])
    return loadings

def main():
    args = parse_arguments()
    
    # Load species profiles
    profile_df = pd.read_csv(args.input, sep='\t', index_col=0)
    
    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(profile_df)
    
    # Perform PCoA
    pcoa_df, explained_variance = perform_pcoa(distance_matrix)
    
    # Calculate feature loadings
    feature_loadings = calculate_feature_loadings(profile_df, pcoa_df)
    
    # Plot PCoA biplot
    plot_pcoa_biplot(pcoa_df, explained_variance, feature_loadings, output=args.output)

if __name__ == "__main__":
    main()

