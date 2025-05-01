#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import distance
from skbio.stats.distance import permanova, permdisp, DistanceMatrix

def load(subject):
    """Load dataset from file or default path."""
    if os.path.isfile(subject):
        df = pd.read_csv(subject, sep='\t', index_col=0)
    else:
        df = pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)
    print(f"Loaded {subject} with shape {df.shape}")
    return df

def save(df, filename, index=True):
    """Save dataframe to output file."""
    output_path = f'../results/{filename}.tsv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def align_samples(dfs, df_cats):
    aligned_dfs = {}

    # Convert boolean columns in df_cats to 0 and 1
    df_cats = df_cats.applymap(lambda x: 1 if isinstance(x, bool) and x else x)
    df_cats = df_cats.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    for key, data in dfs.items():
        # Find common samples between this dataset and df_cats
        common_samples = data.index.intersection(df_cats.index)
        
        # Slice the dataset and df_cats to retain only common samples
        aligned_dfs[key] = data.loc[common_samples]
        print(f"Common samples after alignment for {key}: {len(common_samples)}")

    return aligned_dfs

def explained_variance(dfs, df_cats, from_beta=False, explainors=None, n_iter=30):
    """Perform explained variance analysis using PERMANOVA and PERMDISP."""
    if explainors is None:
        explainors = df_cats.columns.tolist()

    combinations = [(data_key, column) for data_key in dfs.keys() for column in explainors]
    np.random.seed(0)

    output = pd.DataFrame(index=pd.MultiIndex.from_tuples(combinations), columns=['R2', 'anov_pval', 'dispstat', 'disp_pval'])

    for data_key, column in combinations:
        data = dfs[data_key]

        # Only keep samples present in metadata
        common_samples = data.index.intersection(df_cats.index)
        data = data.loc[common_samples]
        meta = df_cats.loc[common_samples]

        if data.shape[0] < 3:
            print(f"⚠️  Skipping {data_key} vs {column} — not enough samples ({data.shape[0]}).")
            continue

        if column not in meta.columns:
            print(f"⚠️  Skipping {data_key} vs {column} — column missing in metadata.")
            continue

        groups = meta[column].dropna().unique()
        if len(groups) < 2:
            print(f"⚠️  Skipping {data_key} vs {column} — less than 2 groups ({groups}).")
            continue

        # Drop non-numeric columns
        data = data.select_dtypes(include=[np.number])

        if data.empty:
            print(f"⚠️  Skipping {data_key} — no numeric data.")
            continue

        try:
            if from_beta:
                dist = DistanceMatrix(data.values, ids=data.index)
            else:
                beta = distance.squareform(distance.pdist(data, metric="braycurtis"))
                dist = DistanceMatrix(beta, ids=data.index)

            anov = permanova(dist, meta, column=column)
            disp = permdisp(dist, meta, column=column)

            output.loc[(data_key, column), :] = [
                anov['test statistic'],
                anov['p-value'],
                disp['test statistic'],
                disp['p-value']
            ]

            print(f"✅ {data_key} vs {column}: R2={anov['test statistic']:.4f}, p={anov['p-value']:.4g}")

        except Exception as e:
            print(f"❌ Error for {data_key} vs {column}: {e}")
            continue

    return output

def parse_args(args):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='explained_variance.py',
        description='Explained Variance using PERMANOVA for beta diversity datasets'
    )
    parser.add_argument('dfs', type=str, nargs='+', help='Dataset(s) for PERMANOVA calculation')
    parser.add_argument('--df_cats', default='meta', type=str, help='Metadata with categories')
    parser.add_argument('-o', '--outfile', type=str, help='Output filename')
    parser.add_argument('-s', '--suffix', type=str, help='Suffix for output')
    parser.add_argument('--from_beta', action='store_true', help='Use precomputed beta-diversity matrix')
    parser.add_argument('--explainor', type=str, help='Single column to explain', default=None)
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])

    # Load your datasets and metadata
    dfs = {df: load(df) for df in args.dfs}
    df_cats = load(args.df_cats)

    # Align samples individually
    aligned_dfs = align_samples(dfs, df_cats)

    # Choose explainors
    explainors = [args.explainor] if args.explainor else None

    # Proceed with the analysis
    output = explained_variance(aligned_dfs, df_cats, from_beta=args.from_beta, explainors=explainors)

    # Save output
    if args.outfile:
        save(output, args.outfile, index=True)
    elif args.suffix:
        base = Path(args.dfs[0]).stem
        save(output, f"{base}_{args.suffix}_perm", index=True)
    else:
        base = Path(args.dfs[0]).stem
        save(output, f"{base}_perm", index=True)

    print("\n Done!")

if __name__ == "__main__":
    main()
