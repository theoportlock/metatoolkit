#!/usr/bin/env python

import argparse
import os
import pandas as pd
import numpy as np
from skbio import TreeNode
from skbio.diversity.alpha import faith_pd

def load_table(table_path):
    # Load the table, transposing so that rows are samples and columns are taxa.
    # Using header=1 if your file has extra header rows.
    df = pd.read_csv(table_path, sep='\t', index_col=0)

    # Only keep columns that contain 't__SGB'
    df = df.loc[:, df.columns.str.contains('t__SGB')]

    # Extract the numeric part from each taxon (e.g. ...|t__SGB10382 → '10382')
    df.columns = df.columns.str.replace(r'.*t__SGB', '', regex=True)

    # Convert all values to numeric (in case of any non-numeric entries)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def load_tree(tree_path):
    """Load a Newick-format phylogenetic tree"""
    return TreeNode.read(tree_path)

def save(df, path):
    if not path.endswith('.tsv'):
        path = f'results/{path}.tsv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t')

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Faith's PD per sample")
    parser.add_argument('table', help='Input abundance table (TSV, samples x taxa)')
    parser.add_argument('tree', help='Newick tree file')
    parser.add_argument('-o', '--outfile', type=str, help='Output file name')
    return parser.parse_args()

def main():
    args = parse_args()
    table = load_table(args.table)
    tree = load_tree(args.tree)

    # Get the set of tip names from the tree
    tip_names = set(tip.name for tip in tree.tips())

    # Calculate intersection and report dropped taxa
    table_taxa = set(table.columns)
    common_taxa = table_taxa.intersection(tip_names)
    dropped = table_taxa - common_taxa
    if dropped:
        print("Dropping taxa not found in tree:", dropped)
        table = table[list(common_taxa)]

    results = {}

    for sample_id, counts in table.iterrows():
        taxa = table.columns.values
        values = counts.values.astype(int)
        pd_value = faith_pd(counts=values, taxa=taxa, tree=tree)
        results[sample_id] = pd_value

    out_df = pd.DataFrame.from_dict(results, orient='index', columns=['Faiths_PD'])
    out_df.index.name = 'sampleID'

    if args.outfile:
        save(out_df, args.outfile)
    else:
        print(out_df)

if __name__ == '__main__':
    main()

