#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from skbio import TreeNode
from skbio.diversity import beta_diversity
from scipy.spatial.distance import pdist, squareform

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_table(table_path, tax_level):
    logging.info(f"Loading abundance table: {table_path}")
    df = pd.read_csv(table_path, sep='\t', index_col=0)
    df = df.loc[:, df.columns.str.contains(fr'{tax_level}')]
    df.columns = df.columns.str.replace(fr'.*{tax_level}SGB', '', regex=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def calculate_standard_metric(metric, table, sample_ids):
    """Calculates non-phylogenetic metrics using scipy."""
    if metric == 'bray-curtis':
        d = squareform(pdist(table.values, metric='braycurtis'))
    elif metric == 'jaccard':
        d = squareform(pdist((table.values > 0).astype(int), metric='jaccard'))
    else: # euclidean
        d = squareform(pdist(table.values, metric='euclidean'))
    return pd.DataFrame(d, index=sample_ids, columns=sample_ids)

def unifrac_worker(chunk_indices, full_table, tree, metric_name):
    """Worker calculates distances for a subset of rows against the full table."""
    sample_ids = full_table.index.tolist()
    subset_ids = [sample_ids[i] for i in chunk_indices]
    skbio_metric = metric_name.replace('-', '_')

    dm = beta_diversity(
        metric=skbio_metric,
        counts=full_table.values,
        ids=sample_ids,
        tree=tree,
        taxa=full_table.columns.values
    )

    df_full = dm.to_data_frame()
    return df_full.loc[subset_ids]

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Beta Diversity (Full Symmetric Output)")
    parser.add_argument('table', help='Input abundance table (TSV)')
    parser.add_argument('-t', '--tree', required=True, help='Newick tree file')
    parser.add_argument('-o', '--outfile', help='Output file name')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads/chunks (default: all cores)')
    parser.add_argument('--tax-level', default='t__', help='Taxonomic prefix (default: t__)')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)

    n_jobs = args.threads if args.threads > 0 else os.cpu_count()
    out_path = args.outfile or "beta_diversity.tsv"

    table = load_table(args.table, args.tax_level)
    tree = TreeNode.read(args.tree)

    # Align table and tree
    tip_names = set(tip.name for tip in tree.tips())
    common_taxa = list(set(table.columns).intersection(tip_names))
    table = table[common_taxa]
    sample_ids = table.index.tolist()

    dist_matrices = {}

    # 1. Standard Metrics
    std_metrics = ['bray-curtis', 'jaccard', 'euclidean']
    logging.info(f"Calculating {std_metrics}...")
    std_results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_standard_metric)(m, table, sample_ids) for m in std_metrics
    )
    for name, df in zip(std_metrics, std_results):
        dist_matrices[name] = df

    # 2. UniFrac Metrics (Chunked Parallelization)
    for m_name in ['weighted-unifrac', 'unweighted-unifrac']:
        logging.info(f"Calculating {m_name} (Parallelized in {n_jobs} chunks)...")
        indices = np.arange(len(sample_ids))
        chunks = np.array_split(indices, n_jobs)

        chunk_results = Parallel(n_jobs=n_jobs)(
            delayed(unifrac_worker)(c, table, tree, m_name)
            for c in tqdm(chunks, desc=f"Processing {m_name}")
        )
        dist_matrices[m_name] = pd.concat(chunk_results)

    # 3. Melt FULL matrix (symmetric output)
    logging.info("Extracting full pairwise distances (symmetric)...")
    final_df = None

    for name, df in dist_matrices.items():
        # Ensure column order matches index order
        df = df.reindex(index=sample_ids, columns=sample_ids)

        # Melt into long format
        long_df = df.stack().reset_index()
        long_df.columns = ['source', 'target', name]

        # Remove self-comparisons (diagonal)
        long_df = long_df[long_df['source'] != long_df['target']]

        if final_df is None:
            final_df = long_df
        else:
            final_df = pd.merge(final_df, long_df, on=['source', 'target'])

    # 4. Save
    logging.info(f"Saving {len(final_df)} pairwise combinations to {out_path}...")

    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    final_df.to_csv(out_path, sep='\t', index=False)
    logging.info("Process finished successfully.")

if __name__ == '__main__':
    main()
