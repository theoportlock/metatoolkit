#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Theo Portlock (modified by ChatGPT)
This script reads an input CSV file, constructs a contingency matrix for two specified columns,
and then either prints or saves the resulting matrix.
"""

import argparse
import pandas as pd
import functions

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Save the contingency matrix of two specified columns from a CSV file."
    )
    parser.add_argument("file", type=str, help="Path to the input CSV file.")
    parser.add_argument("column1", type=str, help="Name of the first column.")
    parser.add_argument("column2", type=str, help="Name of the second column.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Path to save the output contingency matrix")
    return parser.parse_args()

def save_contingency_matrix(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Constructs a contingency matrix (cross-tabulation) for two specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    col1 : str
        The first column name.
    col2 : str
        The second column name.

    Returns
    -------
    pd.DataFrame
        The contingency matrix as a DataFrame.
    """
    # Create the contingency matrix using pandas.crosstab
    contingency_matrix = pd.crosstab(df[col1], df[col2], dropna=False)
    return contingency_matrix

def main():
    args = parse_arguments()

    # Read input CSV file
    df = f.load(args.file)

    # Ensure the specified columns exist in the data
    for col in (args.column1, args.column2):
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in input file.")
            return

    # Create the contingency matrix
    matrix = save_contingency_matrix(df, args.column1, args.column2)

    # Print
    print(matrix)

    # Output the result
    if args.output:
        f.save(matrix, args.output)
    else:
        f.save(out, f'{file}Fisher')

if __name__ == '__main__':
    main()

