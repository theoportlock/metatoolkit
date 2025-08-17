#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from sklearn.decomposition import PCA

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform PCA on a data matrix")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the input data matrix (samples x features, tab-delimited)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file path for PCA results (tab-delimited)")
    parser.add_argument("--n-components", type=int, default=2,
                        help="Number of principal components to compute (default: 2)")
    return parser.parse_args()

def perform_pca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    Perform PCA on the given data matrix.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame of shape (n_samples, n_features), rows are samples.
    n_components : int
        Number of principal components to compute.

    Returns:
    --------
    pd.DataFrame
        A DataFrame of shape (n_samples, n_components) containing the scores
        for each sample on the first n_components PCs, with column names
        including variance explained.
    """
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(df.values)
    var_exp = pca.explained_variance_ratio_

    # Build column labels like "PC1 (45.3%)"
    labels = [
        f"PC{idx+1} ({ratio*100:.1f}%)"
        for idx, ratio in enumerate(var_exp)
    ]
    result = pd.DataFrame(scores, index=df.index, columns=labels)
    return result

def main():
    args = parse_arguments()

    # Read input data
    try:
        df = pd.read_csv(args.input, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error reading input file '{args.input}': {e}")
        return

    # Check that we have more features than components requested
    if args.n_components > min(df.shape):
        print(f"Warning: n_components={args.n_components} is larger than "
              f"min(n_samples, n_features)={min(df.shape)}. "
              f"Reducing n_components to {min(df.shape)}.")
        args.n_components = min(df.shape)

    # Perform PCA
    result = perform_pca(df, args.n_components)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join('results', f"{base}_pca.tsv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    try:
        result.to_csv(output_path, sep='\t')
        print(f"PCA results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving PCA results to '{output_path}': {e}")

if __name__ == "__main__":
    main()

