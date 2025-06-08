#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Group - Groups a dataset')
    parser.add_argument('subject', help="Path to the input data file")
    parser.add_argument('--group_by', nargs='+', help="Columns to group by")
    parser.add_argument('--func', required=True, help="Aggregation function (e.g., sum, mean)")
    parser.add_argument('--axis', type=int, default=0, help="Axis to apply function on, 0 for rows and 1 for columns")
    parser.add_argument('-o', '--output', help="Path to save the output file")
    return parser.parse_args()

def group(df, group_by=None, func='sum', axis=0):
    """
    Group the DataFrame by specified columns or index and apply the aggregation function.

    Parameters:
    - df: DataFrame to group
    - group_by: Columns to group by (None to group by index)
    - func: Aggregation function to apply
    - axis: Axis to apply the function on if not grouping by columns

    Returns:
    - Grouped and aggregated DataFrame
    """
    if group_by:
        outdf = df.groupby(group_by).agg(func)
    else:
        outdf = df.agg(func, axis=axis).to_frame(f'{func}')
    return outdf

def main():
    # Parse arguments
    args = parse_arguments()
    subject = args.subject
    output = args.output or f"results/{subject}_{args.func}.tsv"
    
    # Load data
    df = pd.read_csv(subject, index_col=0, sep='\t')

    # Apply grouping
    out = group(df, group_by=args.group_by, func=args.func, axis=args.axis)
    print(out)

    # Save output
    out.to_csv(output, sep='\t')

if __name__ == "__main__":
    main()

