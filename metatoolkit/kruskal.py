#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
from scipy.stats import kruskal
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='''
    Kruskal-Wallis test for categorical columns with respect to numeric columns.
    ''')
    parser.add_argument('subject', type=str)
    parser.add_argument('df2', type=str, help='Table containing categorical labels')
    args = parser.parse_args()
    return vars(args)

def kruskal_wallis_test(df, df2):
    results = []
    for cat_col in df2.columns:
        if df2[cat_col].nunique() < 2:
            continue  # Skip columns with only one category
        
        for num_col in df.columns:
            valid_idx = df2[cat_col].notna() & df[num_col].notna()
            filtered_df = df[valid_idx]
            filtered_df2 = df2[valid_idx]
            
            groups = [filtered_df[num_col][filtered_df2[cat_col] == category] for category in filtered_df2[cat_col].unique()]
            
            # Ensure all groups have more than one unique value to avoid errors
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                unique_values = [g.nunique() for g in groups]
                if all(u == 1 for u in unique_values):
                    continue  # Skip if all values in all groups are identical
                
                stat, p = kruskal(*groups)
                results.append({'source': cat_col, 'target': num_col, 'statistic': stat, 'p_value': p})
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    known = parse_args()
    subject = known["subject"]
    df2_path = known["df2"]
    
    df = pd.read_csv(subject, index_col=0, sep='\t')
    df2 = pd.read_csv(df2_path, index_col=0, sep='\t')
    
    output = kruskal_wallis_test(df, df2)
    print(output)
    
    output.to_csv(f'results/{subject}_kruskal.tsv', sep='\t')
