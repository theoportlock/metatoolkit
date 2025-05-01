#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from itertools import permutations, product
from pathlib import Path
import functions as f

def parse_args():
    parser = argparse.ArgumentParser(description='Produces Spearman correlations between datasets.')
    parser.add_argument('subject', type=str, help='Primary dataset file path')
    parser.add_argument('df2', type=str, nargs='?', default=None,
                       help='Secondary dataset file path (optional)')
    parser.add_argument('-m', '--mult', action='store_true',
                       help='Apply False Discovery Rate correction')
    args = parser.parse_args()
    return vars(args)

def calculate_correlations(df1, df2=None, fdr=True, min_unique=0):
    # Filter columns with sufficient unique values
    df1 = df1.loc[:, df1.nunique() > min_unique]
    cols1 = df1.columns.tolist()
    
    df2_cols = []
    if df2 is not None:
        df2 = df2.loc[:, df2.nunique() > min_unique]
        df2_cols = df2.columns.tolist()
        pairs = list(product(cols1, df2_cols))
    else:
        pairs = list(permutations(cols1, 2))

    results = []
    for source, target in pairs:
        if df2 is None:
            clean_data = df1[[source, target]].dropna()
        else:
            clean_data = pd.concat([df1[source], df2[target]], axis=1).dropna()
        
        if len(clean_data) < 2:
            cor, pval = (np.nan, np.nan)
        else:
            cor, pval = spearmanr(clean_data[source], clean_data[target])
        
        results.append({
            'source': source,
            'target': target,
            'cor': cor,
            'pval': pval if not np.isnan(pval) else 1.0
        })

    results_df = pd.DataFrame(results)
    
    if fdr and not results_df.empty:
        pvals = results_df['pval'].values
        _, qvals = fdrcorrection(pvals)
        results_df['qval'] = qvals
    else:
        results_df['qval'] = np.nan

    return results_df[['source', 'target', 'cor', 'pval', 'qval']]

def main():
    args = parse_args()
    primary_path = args['subject']
    secondary_path = args['df2']
    use_fdr = args['mult']
    
    primary_df = f.load(primary_path)
    secondary_df = f.load(secondary_path) if secondary_path else None
    
    output = calculate_correlations(primary_df, secondary_df, fdr=use_fdr)
    
    # Generate output name
    base_name = Path(primary_path).stem
    if secondary_path:
        base_name += f"_vs_{Path(secondary_path).stem}"
    
    print(output)
    f.save(output, f"{base_name}_spearman_corr")

if __name__ == "__main__":
    main()
