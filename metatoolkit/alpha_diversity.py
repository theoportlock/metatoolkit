#!/usr/bin/env python

import argparse
import os
import pandas as pd
import numpy as np
from skbio import TreeNode
from skbio.diversity.alpha import shannon, faith_pd


def load_table(table_path, tax_level):
    """
    Load the abundance table (TSV).
    Rows = samples, columns = taxa.
    Keeps only columns containing the chosen taxonomic prefix (e.g. t__ or s__).
    Extracts the numeric/ID suffix after the prefix.
    """
    df = pd.read_csv(table_path, sep='\t', index_col=0)

    # Only keep taxa columns of interest
    df = df.loc[:, df.columns.str.contains(fr'{tax_level}')]

    # Extract numeric SGB ID (remove everything before the prefix)
    df.columns = df.columns.str.replace(fr'.*{tax_level}SGB', '', regex=True)

    # Ensure numeric values
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def load_tree(tree_path):
    """Load a Newick-format phylogenetic tree."""
    return TreeNode.read(tree_path)


def save(df, path):
    """Save the dataframe as TSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate alpha diversity metrics (Shannon, Richness, Faith's PD) per sample."
    )
    parser.add_argument('table', help='Input abundance table (TSV, samples x taxa)')
    parser.add_argument('-t','--tree', help='Newick tree file (required only for Faith\'s PD)')
    parser.add_argument(
        '-o', '--outfile', type=str,
        help='Output file name (default: alpha_diversity.tsv in same directory as input table)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['shannon', 'richness', 'faiths', 'all'],
        default=['all'],
        help='Which alpha diversity metrics to calculate (default: all).'
    )
    parser.add_argument(
        '--tax-level',
        #choices=['t__', 's__'],
        default='t__',
        help='Taxonomic prefix of interest in column names (default: t__).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If output path not provided, put it in the same directory as the input table
    if args.outfile:
        out_path = args.outfile
    else:
        in_dir = os.path.dirname(os.path.abspath(args.table))
        out_path = os.path.join(in_dir, 'alpha_diversity.tsv')

    table = load_table(args.table, args.tax_level)

    # Decide which metrics to compute
    metrics = args.metrics
    if 'all' in metrics:
        metrics = ['shannon', 'richness', 'faiths']

    # Prepare tree and filtered table ONLY for Faith's PD
    tree = None
    table_for_faith = None
    if 'faiths' in metrics:
        tree = load_tree(args.tree)
        tip_names = set(tip.name for tip in tree.tips())
        common_taxa = set(table.columns).intersection(tip_names)
        dropped = set(table.columns) - common_taxa
        if dropped:
            print("Dropping taxa not found in tree:", dropped)
        table_for_faith = table[list(common_taxa)]

    results = {}

    for sample_id, counts in table.iterrows():
        values = counts.values
        row = {}
        if 'shannon' in metrics:
            row['Shannon'] = shannon(values)
        if 'richness' in metrics:
            row['Richness'] = np.count_nonzero(values > 0)
        if 'faiths' in metrics:
            if sample_id in table_for_faith.index:
                faith_values = table_for_faith.loc[sample_id].values
                faith_taxa = table_for_faith.columns.values
                row['Faiths_PD'] = faith_pd(counts=faith_values, taxa=faith_taxa, tree=tree)
            else:
                row['Faiths_PD'] = np.nan
        results[sample_id] = row

    out_df = pd.DataFrame.from_dict(results, orient='index')
    out_df.index.name = 'sampleID'

    save(out_df, out_path)
    print(f"Alpha diversity table saved to: {out_path}")


if __name__ == '__main__':
    main()

