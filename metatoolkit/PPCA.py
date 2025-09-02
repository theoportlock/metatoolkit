#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from ppca import PPCA

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform PPCA on a data matrix with missing values")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input data matrix (samples x features, tab-delimited)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file path for PPCA results (tab-delimited)")
    parser.add_argument("--n-components", type=int, default=2,
                        help="Number of principal components to compute (default: 2)")
    return parser.parse_args()

def perform_ppca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """
    Perform PPCA (tolerant to missing values) using EM algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Data matrix with rows as samples, columns as features. Can contain NaNs.
    n_components : int
        Number of principal components to compute.

    Returns
    -------
    pd.DataFrame
        Transformed sample scores in PC space.
    """
    model = PPCA()
    model.fit(df.values, d=n_components)
    scores = model.transform()
    labels = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(scores, index=df.index, columns=labels)

def main():
    args = parse_arguments()

    try:
        df = pd.read_csv(args.input, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error reading input file '{args.input}': {e}")
        return

    if args.n_components > min(df.shape):
        print(f"Warning: n_components={args.n_components} is larger than "
              f"min(n_samples, n_features)={min(df.shape)}. Reducing it.")
        args.n_components = min(df.shape)

    result = perform_ppca(df, args.n_components)

    output_path = args.output or os.path.join(
        'results',
        f"{os.path.splitext(os.path.basename(args.input))[0]}_ppca.tsv"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result.to_csv(output_path, sep='\t')
        print(f"PPCA results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving PPCA results to '{output_path}': {e}")

if __name__ == "__main__":
    main()

