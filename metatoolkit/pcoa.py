#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import skbio

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform PCoA on a distance matrix")
    parser.add_argument("-i", "--input", type=str, help="Distance matrix path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    return parser.parse_args()

def perform_pcoa(df):
    """
    Perform PCoA on the given distance matrix DataFrame.
    Parameters:
    df (pd.DataFrame): A square DataFrame representing a distance matrix, where
                       rows and columns are samples, and values are distances.
    Returns:
    pd.DataFrame: A DataFrame with the first two PCoA components (PCo1 and PCo2) for each sample.
    Example:
    --------
    Test data input:
    Consider the following sample distance matrix:
        A      B      C
    A  0.0    0.1    0.3
    B  0.1    0.0    0.4
    C  0.3    0.4    0.0
    Create the DataFrame and run PCoA:
    >>> import pandas as pd
    >>> import skbio
    >>> data = {'A': [0.0, 0.1, 0.3], 'B': [0.1, 0.0, 0.4], 'C': [0.3, 0.4, 0.0]}
    >>> df = pd.DataFrame(data, index=['A', 'B', 'C'])
    >>> perform_pcoa(df)
       PCo1 (100.0%)  PCo2 (0.0%)
    A      -0.066667          0.0
    B      -0.166667         -0.0
    C       0.233333          0.0
    """
    DM_dist = skbio.stats.distance.DistanceMatrix(df)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    label = PCoA.proportion_explained.apply(" ({:.1%})".format)
    results = PCoA.samples.copy()
    result = pd.DataFrame(index=df.index)
    result["PCo1" + label.loc["PC1"]] = results.iloc[:, 0].values
    result["PCo2" + label.loc["PC2"]] = results.iloc[:, 1].values
    return result

def main():
    """Main function to execute the script."""
    args = parse_arguments()
    subject = args.input

    # Read input data
    try:
        df = pd.read_csv(subject, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error reading input file {subject}: {e}")
        return

    # Perform PCoA
    result = perform_pcoa(df)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default output path
        base_name = os.path.splitext(os.path.basename(subject))[0]
        output_path = f'../results/{base_name}_pcoa.tsv'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results to file
    try:
        result.to_csv(output_path, sep='\t')
        print(f"PCoA results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")

if __name__ == "__main__":
    main()
