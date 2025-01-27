#!/usr/bin/env python
import pandas as pd
import skbio
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate and plot a PCoA biplot from metagenomic species profiles.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Metagenomic species profile file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path for the plot image")
    return parser.parse_args()

def calculate_distance_matrix(profile_df):
    """Calculate a Bray-Curtis distance matrix from metagenomic species profiles."""
    distance_matrix = skbio.diversity.beta_diversity("braycurtis", profile_df.T.values, ids=profile_df.columns)
    return distance_matrix

def perform_pcoa(distance_matrix):
    """Perform PCoA on the given distance matrix."""
    PCoA = skbio.stats.ordination.pcoa(distance_matrix, number_of_dimensions=2)
    explained_variance = PCoA.proportion_explained
    return PCoA.samples, explained_variance

def calculate_feature_loadings(profile_df, pcoa_df):
    """Calculate feature loadings by correlating each feature with the PCoA axes."""
    loadings = pd.DataFrame(index=profile_df.index)
    for axis in range(2):
        loadings[f"PCo{axis + 1}"] = profile_df.corrwith(pcoa_df.iloc[:, axis], axis=1)
    return loadings

def plot_pcoa_biplot(pcoa_df, explained_variance, feature_loadings, output_path=None):
    """Plot a PCoA biplot with samples and top 5 feature loadings for each axis."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pcoa_df.iloc[:, 0], pcoa_df.iloc[:, 1], color='blue', label='Samples')

    # Get the top 5 features with the strongest loading values for each axis
    top_features = pd.concat([
        feature_loadings['PCo1'].abs().nlargest(5),
        feature_loadings['PCo2'].abs().nlargest(5)
    ]).index.unique()

    # Plot feature loadings as arrows
    for feature in top_features:
        ax.arrow(0, 0, feature_loadings.loc[feature, "PCo1"] * 0.1, feature_loadings.loc[feature, "PCo2"] * 0.1, 
                 color='red', alpha=0.5)
        ax.text(feature_loadings.loc[feature, "PCo1"] * 0.12, feature_loadings.loc[feature, "PCo2"] * 0.12, 
                feature, color='red', ha='center', va='center', fontsize=8)

    ax.set_xlabel(f'PCo1 ({explained_variance[0]:.1%})')
    ax.set_ylabel(f'PCo2 ({explained_variance[1]:.1%})')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.legend()
    ax.set_title('PCoA Biplot with Sample Scores and Top Feature Loadings')

    # Save plot if an output path is specified
    if output_path:
        plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load the metagenomic species profile
    data = pd.read_csv(args.input, sep='\t', index_col=0)
    
    # Run the analysis pipeline
    distance_matrix = calculate_distance_matrix(data)
    pcoa_df, explained_variance = perform_pcoa(distance_matrix)
    feature_loadings = calculate_feature_loadings(data, pcoa_df)
    
    # Plot the results
    plot_pcoa_biplot(pcoa_df, explained_variance, feature_loadings, output_path=args.output)
