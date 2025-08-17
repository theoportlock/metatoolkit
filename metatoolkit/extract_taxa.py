#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Theo Portlock
Generalized script to extract taxa at a specified level from MetaPhlAn output,
with optional parent prefix and removal of taxonomic prefixes (e.g., 's__', 't__').
"""

import pandas as pd
import argparse
import re

def strip_prefix(name):
    """Remove taxonomic prefix (e.g. 's__', 't__') from a taxon name"""
    return re.sub(r'^[a-z]__', '', name)

def extract_taxa(df, tax_level, keep_parent=False):
    # Match columns that contain the specified level
    pattern = r'\|' + re.escape(tax_level)
    matching_cols = df.columns[df.columns.str.contains(pattern)]
    df = df[matching_cols]

    if keep_parent:
        # Extract the parent and child names without prefixes
        df.columns = (
            df.columns.str.extract(
                r'\|([a-z]__[^|]+)\|' + re.escape(tax_level) + r'([^|]+)$', expand=True
            )
            .agg('|'.join, axis=1)
            .map(strip_prefix)  # strip the prefix of the parent
        )
    else:
        # Only extract the specified level (e.g. t__)
        df.columns = df.columns.str.extract(r'\|' + re.escape(tax_level) + r'([^|]+)$', expand=False)

    # Always strip the taxonomic prefix from the child name too
    df.columns = df.columns.map(strip_prefix)
    df.columns.name = tax_level.rstrip('_')
    return df

def main():
    parser = argparse.ArgumentParser(description='Extract taxa from MetaPhlAn table.')
    parser.add_argument('input', help='Input MetaPhlAn TSV file')
    parser.add_argument('-l', '--level', required=True,
                        choices=['k__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__', 't__'],
                        help='Taxonomic level to extract (e.g., g__, s__, t__)')
    parser.add_argument('-o', '--output', help='Output TSV file (default: auto-named)', default=None)
    parser.add_argument('--keep-parent', action='store_true',
                        help='Retain the immediate parent name in column labels (no taxonomic prefixes)')

    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\t', index_col=0)

    df_out = extract_taxa(df, args.level, keep_parent=args.keep_parent)

    output_file = args.output or re.sub(r'\.tsv$', f'_{args.level.rstrip("_")}.tsv', args.input)
    df_out.to_csv(output_file, sep='\t')
    print(f'Saved: {output_file}')

if __name__ == '__main__':
    main()

