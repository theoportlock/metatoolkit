#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    if not subject.endswith('.tsv'):
        subject = f'results/{subject}.tsv'
    output_path = subject
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def filter(df, **kwargs):
    df.index = df.index.astype(str)
    if kwargs.get('filter_df') is not None:
        filter_df = kwargs.get('filter_df')
        if filter_df is not None:
            if kwargs.get('filter_df_axis') == 1:
                common_cols = df.columns.intersection(filter_df.index)
                df = df.loc[:, common_cols]
            else:
                common_rows = df.index.intersection(filter_df.index)
                df = df.loc[common_rows]
    if kwargs.get('colfilt'):
        df = df.loc[:, df.columns.str.contains(kwargs.get('colfilt'), regex=True)]
    if kwargs.get('rowfilt'):
        df = df.loc[df.index.str.contains(kwargs.get('rowfilt'), regex=True)]
    if kwargs.get('prevail'):
        df = df.loc[:, df.agg(np.count_nonzero, axis=0).gt(df.shape[0]*kwargs.get('prevail'))]
    if kwargs.get('abund'):
        df = df.loc[:, df.mean().gt(kwargs.get('abund'))]
    if kwargs.get('min_unique'):
        df = df.loc[:, df.nunique().gt(kwargs.get('min_unique'))]
    if kwargs.get('nonzero'):
        df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    if kwargs.get('min_nonzero_rows') is not None:
        df = df[df.astype(bool).sum(axis=1) >= kwargs.get('min_nonzero_rows')]
    if kwargs.get('min_nonzero_cols') is not None:
        df = df.loc[:, df.astype(bool).sum(axis=0) >= kwargs.get('min_nonzero_cols')]
    if kwargs.get('numeric_only'):
        if ~(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())).any():
            df = pd.DataFrame()
    queries = kwargs.get('query')
    if queries:
        combined_query = ' & '.join(f"({q})" for q in queries)
        df = df.query(combined_query)
    if kwargs.get('dtype'):
        df = df.select_dtypes(kwargs.get('dtype'))
    if df.empty:
        return None
    else:
        return df

def parse_args():
    parser = argparse.ArgumentParser(description='Filter a TSV file based on various criteria.')
    parser.add_argument('subject', help='Input file or subject name')
    parser.add_argument('-rf', '--rowfilt', type=str, help='Regex for index filtering')
    parser.add_argument('-cf', '--colfilt', type=str, help='Regex for column filtering')
    parser.add_argument('-q', '--query', action='append', help='Pandas custom query (can be used multiple times)')
    parser.add_argument('-m', '--min_unique', type=int, help='Minimum number of unique values')
    parser.add_argument('-fdf', '--filter_df', help='CSV file for filtering indices')
    parser.add_argument('-fdfx', '--filter_df_axis', type=int, help='Axis to filter the dataframe indices (0 or 1)')
    parser.add_argument('-absgt', type=float, help='Absolute greater than threshold')
    parser.add_argument('-p', '--prevail', type=float, help='Prevalence threshold')
    parser.add_argument('-a', '--abund', type=float, help='Abundance threshold')
    parser.add_argument('-o', '--outfile', type=str, help='Output file name')
    parser.add_argument('-s', '--suffix', type=str, help='Suffix to append to the subject for output')
    parser.add_argument('--numeric_only', action='store_true', help='Select numeric columns only')
    parser.add_argument('--nonzero', action='store_true', help='Remove rows and columns that sum to zero')
    parser.add_argument('--min_nonzero_rows', type=int, help='Minimum number of non-zero values required in a row')
    parser.add_argument('--min_nonzero_cols', type=int, help='Minimum number of non-zero values required in a column')
    parser.add_argument('--print_counts', action='store_true', help='Print the number of rows and columns that have been filtered')
    parser.add_argument('-dt', '--dtype', type=str, help='Select columns with a specific dtype')
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}

def main():
    known = parse_args()
    subject = known.get("subject")
    known.pop('subject')

    original_df = load(subject)

    if subject is not None and os.path.isfile(subject):
        subject = Path(subject).stem

    if known.get("filter_df"):
        known['filter_df'] = load(known.get("filter_df"))

    output = filter(original_df.copy(), **known)

    if known.get("print_counts"):
        if output is None:
            print("All rows and columns have been filtered out.")
        else:
            original_rows, original_cols = original_df.shape
            filtered_rows, filtered_cols = output.shape
            print(f"Rows filtered: {original_rows - filtered_rows} out of {original_rows}")
            print(f"Columns filtered: {original_cols - filtered_cols} out of {original_cols}")

    print(output)

    if output is not None:
        if known.get("outfile"):
            save(output, known.get("outfile"))
        elif known.get("suffix"):
            subj = str(subject) if subject is not None else ""
            suffix = str(known.get("suffix")) if known.get("suffix") is not None else ""
            save(output, subj + suffix)
        else:
            subj = subject if subject is not None else ""
            save(output, subj + 'filter')

if __name__ == "__main__":
    main()

