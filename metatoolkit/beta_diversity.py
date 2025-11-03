#!/usr/bin/env python

import argparse
import os
import pandas as pd
import numpy as np
from itertools import combinations

from skbio import TreeNode
from skbio.diversity import beta_diversity
from scipy.spatial.distance import pdist, squareform   # only for bray & jaccard


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
    df.to_csv(path, sep='\t', index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Calculate pairwise beta diversity metrics "
                     "(Bray–Curtis, Jaccard, Weighted/Unweighted UniFrac).")
    )
    parser.add_argument('table', help='Input abundance table (TSV, samples x taxa)')
    parser.add_argument('-t','--tree', help='Newick tree file (required only for UniFrac)')
    parser.add_argument(
        '-o', '--outfile', type=str,
        help='Output file name (default: beta_diversity.tsv in same directory as input table)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        choices=['bray-curtis', 'jaccard', 'weighted-unifrac', 'unweighted-unifrac', 'all'],
        default=['all'],
        help='Which beta diversity metrics to calculate (default: all).'
    )
    parser.add_argument(
        '--tax-level',
        default='t__',
        help='Taxonomic prefix of interest in column names (default: t__).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Output path
    if args.outfile:
        out_path = args.outfile
    else:
        in_dir = os.path.dirname(os.path.abspath(args.table))
        out_path = os.path.join(in_dir, 'beta_diversity.tsv')

    table = load_table(args.table, args.tax_level)
    sample_ids = table.index.tolist()
    metrics = args.metrics
    if 'all' in metrics:
        metrics = ['bray-curtis', 'jaccard', 'weighted-unifrac', 'unweighted-unifrac']

    # Prepare tree only if needed for UniFrac
    tree = None
    if any(m in metrics for m in ['weighted-unifrac', 'unweighted-unifrac']):
        if not args.tree:
            raise ValueError("A Newick tree (--tree) is required for UniFrac metrics.")
        tree = load_tree(args.tree)

        # Keep only taxa present in both table and tree tips
        tip_names = set(tip.name for tip in tree.tips())
        common_taxa = set(table.columns).intersection(tip_names)
        dropped = set(table.columns) - common_taxa
        if dropped:
            print("Dropping taxa not found in tree:", dropped)
        table = table[list(common_taxa)]

    # Precompute distance matrices
    dist_matrices = {}

    if 'bray-curtis' in metrics:
        bc = squareform(pdist(table.values, metric='braycurtis'))
        dist_matrices['bray-curtis'] = pd.DataFrame(bc, index=sample_ids, columns=sample_ids)

    if 'jaccard' in metrics:
        presence = (table.values > 0).astype(int)
        jc = squareform(pdist(presence, metric='jaccard'))
        dist_matrices['jaccard'] = pd.DataFrame(jc, index=sample_ids, columns=sample_ids)

    if 'weighted-unifrac' in metrics:
        dist_matrices['weighted-unifrac'] = beta_diversity(
            metric='weighted_unifrac',
            counts=table.values,
            ids=sample_ids,
            tree=tree,
            taxa=table.columns.values   # <-- required!
        ).to_data_frame()

    if 'unweighted-unifrac' in metrics:
        presence = (table.values > 0).astype(int)
        dist_matrices['unweighted-unifrac'] = beta_diversity(
            metric='unweighted_unifrac',
            counts=presence,
            ids=sample_ids,
            tree=tree,
            taxa=table.columns.values   # <-- required!
        ).to_data_frame()

    # Build long-format table
    rows = []
    for i, j in combinations(sample_ids, 2):
        row = {'source': i, 'target': j}
        for m in metrics:
            row[m] = dist_matrices[m].loc[i, j]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    save(out_df, out_path)
    print(f"Beta diversity table saved to: {out_path}")


if __name__ == '__main__':
    main()

