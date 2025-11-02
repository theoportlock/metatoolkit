#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Group - Groups a dataset')
    parser.add_argument('subject', help="Path to the input data file or subject name")
    parser.add_argument(
        '--group_by',
        nargs='+',
        help="Columns to group by (+ nargs), or 'all' to aggregate all samples together"
    )
    parser.add_argument(
        '--func',
        nargs='+',
        required=True,
        help="Aggregation function(s), e.g. mean median sum"
    )
    parser.add_argument('--axis', type=int, default=0, help="Axis to apply function on, 0 for rows and 1 for columns")
    parser.add_argument('-o', '--output', help="Path to save the output file")
    parser.add_argument(
        '--meta',
        help="Path to metadata file to inner-join with subject data before grouping"
    )
    return parser.parse_args()


def load_data(path_or_name):
    path = Path(path_or_name)
    if path.is_file():
        return pd.read_csv(path, sep='\t', index_col=0)
    else:
        return pd.read_csv(Path('results') / f'{path_or_name}.tsv', sep='\t', index_col=0)


def merge_meta(df, meta_path, group_by):
    mdf = pd.read_csv(meta_path, sep='\t', index_col=0)

    # Only join metadata if we’re actually grouping by something
    if group_by and (len(group_by) != 1 or group_by[0].lower() != 'all'):
        missing = [col for col in group_by if col not in mdf.columns]
        if missing:
            raise ValueError(f"Metadata file missing required columns: {missing}")
        mdf = mdf[group_by]
        df = df.join(mdf, how='inner')

    return df


def flatten_columns(df):
    """Flatten MultiIndex columns to 'col_func' form if needed."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["{}_{}".format(col, func) for col, func in df.columns]
    return df


def group(df, group_by=None, funcs=None, axis=0):
    """
    Group the DataFrame by specified columns or aggregate everything if 'all' is given.
    Supports multiple aggregation functions and consistent output formatting.
    """
    if funcs is None:
        funcs = ['sum']

    # --- Case 1: aggregate all samples together (no stratification) ---
    if group_by and len(group_by) == 1 and group_by[0].lower() == 'all':
        outdf = df.agg(funcs, axis=axis)

        # If multiple funcs, flatten into one row
        if isinstance(outdf, pd.DataFrame):
            # outdf looks like:
            #        Mreads
            # mean   61.96
            # std    10.71
            outdf = outdf.T
            if isinstance(outdf.columns, pd.MultiIndex):
                outdf.columns = [f"{col}_{func}" for func, col in outdf.columns.to_flat_index()]
            else:
                outdf.columns = [f"{col}_{func}" for col, func in zip(outdf.columns, funcs)]
            outdf.index = ['all']
        else:
            # Single function
            outdf = pd.DataFrame(outdf).T
            outdf.index = ['all']
        return outdf

    # --- Case 2: group by specific columns ---
    if group_by:
        outdf = df.groupby(group_by).agg(funcs)
        if isinstance(outdf.columns, pd.MultiIndex):
            outdf.columns = [f"{col}_{func}" for col, func in outdf.columns.to_flat_index()]
    else:
        # No grouping, just aggregate over axis
        outdf = df.agg(funcs, axis=axis)
        if isinstance(outdf, pd.DataFrame):
            outdf = outdf.T
            if isinstance(outdf.columns, pd.MultiIndex):
                outdf.columns = [f"{col}_{func}" for func, col in outdf.columns.to_flat_index()]
            else:
                outdf.columns = [f"{col}_{func}" for col, func in zip(outdf.columns, funcs)]
        else:
            outdf = pd.DataFrame(outdf).T
        outdf.index = ['all']

    return outdf


def main():
    args = parse_arguments()
    subject = args.subject

    # Determine output filename
    func_str = "_".join(args.func)
    outname = args.output or f"results/{Path(subject).stem}_{func_str}.tsv"
    outpath = Path(outname)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Load subject data
    df = load_data(subject)

    # Merge metadata if specified
    if args.meta:
        df = merge_meta(df, args.meta, args.group_by)

    # Apply grouping
    out = group(df, group_by=args.group_by, funcs=args.func, axis=args.axis)

    # Print and save
    print(out)
    out.to_csv(outpath, sep='\t')


if __name__ == "__main__":
    main()

