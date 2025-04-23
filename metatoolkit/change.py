#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


def parse_arguments():
    parser = argparse.ArgumentParser(description='''
    Change - Bivariate analysis of feature changes
    ''')
    parser.add_argument('subject', help='Path to dataset file or subject name')
    parser.add_argument(
        '-df2', '--df2',
        required=True,
        help='Path to metadata/one-hot-encoded file or subject name'
    )
    parser.add_argument(
        '-c', '--columns',
        nargs='+',
        help='Which columns of df2 to use for splitting (default: all)'
    )
    parser.add_argument(
        '-a', '--analysis',
        nargs='+',
        choices=['mww', 'fc', 'diffmean', 'summary'],
        default=['mww','fc','diffmean','summary'],
        help='Methods of analysis to perform'
    )
    return parser.parse_args()


def load_table(path_or_name: str) -> (pd.DataFrame, str):
    """Load a TSV from file or ../results/{name}.tsv; return (df, stem)."""
    p = Path(path_or_name)
    if p.is_file():
        stem = p.stem
        df = pd.read_csv(p, sep='\t', index_col=0)
    else:
        stem = path_or_name
        df = pd.read_csv(Path('../results') / f'{stem}.tsv', sep='\t', index_col=0)
    return df, stem


def save(df: pd.DataFrame, stem: str, index: bool = True):
    """Save df to ../results/{stem}.tsv."""
    out = Path('../results') / f'{stem}.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=index)
    print(f"Saved results to {out}")


def splitter(df: pd.DataFrame,
             df2: pd.DataFrame,
             column: str) -> dict:
    """
    For each unique level in df2[column], inner-join df with those rows,
    dropping the split column. Returns {level: sub-DataFrame}.
    """
    output = {}
    if df2 is None:
        df2 = df.copy()
    for level in df2[column].unique():
        # select rows of df2 matching this level
        idx = df2[df2[column] == level].index
        # join only those rows, drop the split column
        merged = (
            df.loc[idx, df.columns.difference([column])]
              .copy()
        )
        output[level] = merged
    return output


def mww(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    pvals = [mannwhitneyu(df1[col], df2[col]).pvalue for col in df1.columns]
    out = pd.DataFrame(pvals, index=df1.columns, columns=['MWW_pval'])
    out['MWW_qval'] = fdrcorrection(out['MWW_pval'])[1]
    return out


def fc(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    mean1 = df1.mean()
    mean2 = df2.mean()
    fc = mean1.div(mean2)
    out = pd.DataFrame(fc, columns=['FC'])
    out['Log2FC'] = np.log2(out['FC'])
    out['Log10FC'] = np.log10(out['FC'])
    return out


def diffmean(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    diff = df1.mean().sub(df2.mean())
    return pd.DataFrame(diff, columns=['diffmean'])


def summary(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    def get_sum(df: pd.DataFrame) -> pd.Series:
        m = df.mean().round(2).astype(str)
        s = df.std().round(2).astype(str)
        return m + ' ± ' + s

    return pd.DataFrame({
        'source_true_summary': get_sum(df1),
        'source_false_summary': get_sum(df2)
    })


def change(df: pd.DataFrame,
           df2: pd.DataFrame,
           columns: list = None,
           analyses: list = None) -> pd.DataFrame:
    """
    For each split column and each pair of levels within it, perform analyses.
    Returns a MultiIndexed DataFrame indexed by (column, level1_vs_level2, feature).
    """
    if columns is None:
        columns = df2.columns.tolist()

    available = {'mww': mww, 'fc': fc, 'diffmean': diffmean, 'summary': summary}
    all_results = []

    for col in columns:
        levels = df2[col].unique()
        # for every pair of distinct levels
        for i, lvl1 in enumerate(levels):
            for lvl2 in levels[i+1:]:
                sub = splitter(df, df2, col)
                df1, df2_ = sub[lvl1], sub[lvl2]
                if df1.empty or df2_.empty:
                    continue

                # run each analysis
                dfs = []
                for method in analyses:
                    dfs.append(available[method](df1, df2_))

                # concat side by side
                combined = pd.concat(dfs, axis=1)
                # label the index
                combined.index.set_names('feature', inplace=True)
                # add MultiIndex: (column, comparison)
                combined.columns  # features are columns; index is features
                combined = combined.assign(
                    source=col,
                    comparison=f'{lvl1}_vs_{lvl2}'
                ).set_index(['source','comparison'], append=True)
                all_results.append(combined)

    if not all_results:
        return pd.DataFrame()  # empty

    # combine all, swap levels so (source,comparison,feature)
    result = pd.concat(all_results)
    result = result.reorder_levels(['source','comparison','feature'])
    return result


def main():
    args = parse_arguments()
    df, stem1 = load_table(args.subject)
    df2, stem2 = load_table(args.df2)

    out = change(df, df2, columns=args.columns, analyses=args.analysis)
    print(out)
    save(out, f'{stem1}_change')


if __name__ == '__main__':
    main()
