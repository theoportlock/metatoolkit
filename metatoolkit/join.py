#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import metatoolkit.functions as f
import argparse
import pandas as pd

def combine_identical_columns(df, suffixes):
    """
    Combine columns with identical values that received suffixes during merging.
    Preserves the original column name if both suffixed versions are identical.
    """
    suffix1, suffix2 = suffixes
    base_columns = {}
    
    # Identify potential columns to combine
    for col in df.columns:
        if col.endswith(suffix1):
            base = col[:-len(suffix1)]
            base_columns.setdefault(base, {})[suffix1] = col
        elif col.endswith(suffix2):
            base = col[:-len(suffix2)]
            base_columns.setdefault(base, {})[suffix2] = col

    # Process matching column pairs
    to_drop = []
    to_add = {}
    
    for base, cols in base_columns.items():
        if suffix1 in cols and suffix2 in cols:
            col1 = cols[suffix1]
            col2 = cols[suffix2]
            
            if df[col1].equals(df[col2]):
                to_add[base] = df[col1]
                to_drop.extend([col1, col2])

    return df.drop(columns=to_drop).assign(**to_add)

def main():
    parser = argparse.ArgumentParser(description='Join two dataframes using pandas merge functionality.')
    parser.add_argument('file1', type=str, help='Path to the first input file.')
    parser.add_argument('file2', type=str, help='Path to the second input file.')
    parser.add_argument('-o', '--output', type=str, default='joined', help='Output file name prefix.')
    parser.add_argument('--how', type=str, default='inner', 
                      choices=['inner', 'outer', 'left', 'right', 'cross'],
                      help='Type of merge to be performed.')
    parser.add_argument('--on', type=str, help='Common column name(s) (comma-separated) to join on.')
    parser.add_argument('--left_on', type=str, help='Column name(s) from left dataframe (comma-separated).')
    parser.add_argument('--right_on', type=str, help='Column name(s) from right dataframe (comma-separated).')
    parser.add_argument('--suffixes', type=str, default='_x,_y',
                      help='Suffixes to apply to overlapping columns (comma-separated).')
    
    args = parser.parse_args()

    # Load dataframes
    #df1 = pd.read_csv('../results/'+args.file1+'.tsv', sep='\t')
    #df2 = pd.read_csv('../results/'+args.file2+'.tsv', sep='\t')
    df1 = f.load(args.file1).reset_index()
    df2 = f.load(args.file2).reset_index()

    # Process suffixes
    suffixes = args.suffixes.split(',')
    if len(suffixes) != 2:
        raise ValueError("Must provide exactly two suffixes separated by a comma")

    # Determine merge parameters
    merge_params = {}
    if args.on:
        merge_params['on'] = args.on.split(',')
    elif args.left_on and args.right_on:
        merge_params['left_on'] = args.left_on.split(',')
        merge_params['right_on'] = args.right_on.split(',')
    else:
        merge_params['left_index'] = True
        merge_params['right_index'] = True

    # Perform merge
    merged_df = pd.merge(
        df1,
        df2,
        how=args.how,
        suffixes=suffixes,
        **merge_params
    )

    # Combine identical columns that received suffixes
    merged_df = combine_identical_columns(merged_df, suffixes)

    # Set index as first column
    merged_df = merged_df.set_index(merged_df.columns[0])

    # Save results
    f.save(merged_df, args.output)

if __name__ == '__main__':
    main()
