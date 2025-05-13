#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Theo Portlock (modified by ChatGPT)
This script calculates the Chi Squared test and Cramér's V for all pairs of categorical columns in a DataFrame.
"""

import argparse
import pandas as pd
import numpy as np
from itertools import permutations
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from statsmodels.stats.multitest import fdrcorrection

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    output_path = f'results/{subject}.tsv' 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Chi Squared test and Cramér's V for all pairs of categorical columns in a DataFrame.")
    parser.add_argument("file", type=str, nargs="?", help="Path to the input file (CSV format). If not provided, example data is used.")
    parser.add_argument("-o", "--output", type=str, help="Path to save the output file (CSV format).", default=None)
    return parser.parse_args()

def chi_squared(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Chi Squared statistic, p-value, degrees of freedom, and Cramér's V for all pairs of columns in a DataFrame.
    This version works with contingency tables of any shape.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where columns are categorical variables and rows are observations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with a MultiIndex of column pairs, containing the Chi Squared statistic, p-value, degrees of freedom,
        FDR-adjusted q-values, and Cramér's V.

    Example
    -------
    >>> data = {
    ...     'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red'],
    ...     'Shape': ['Circle', 'Square', 'Triangle', 'Square', 'Circle', 'Triangle', 'Triangle', 'Circle', 'Square', 'Circle'],
    ...     'Size':  ['Small', 'Medium', 'Large', 'Large', 'Small', 'Medium', 'Medium', 'Large', 'Small', 'Medium']
    ... }
    >>> df = pd.DataFrame(data)
    >>> chi_squared(df)
    """
    column_pairs = list(permutations(df.columns, 2))
    results = {
        'chi2': [],
        'pval': [],
        'dof': [],
        'cramers_v': []
    }

    for source, target in column_pairs:
        contingency_table = pd.crosstab(df[target], df[source])
        chi2_stat, pvalue, dof, expected = chi2_contingency(contingency_table)
        v = association(contingency_table, method="cramer")
        results['chi2'].append(chi2_stat)
        results['pval'].append(pvalue)
        results['dof'].append(dof)
        results['cramers_v'].append(v)

    index = pd.MultiIndex.from_tuples(column_pairs, names=['source', 'target'])
    result_df = pd.DataFrame(results, index=index)
    result_df['qval'] = fdrcorrection(result_df.pval)[1]
    return result_df

def main():
    args = parse_arguments()

    df = load(args.file)

    # Calculate Chi Squared test results
    result = chi_squared(df)
    print(result)
    save(result, args.file + '_chisq')

if __name__ == '__main__':
    main()
