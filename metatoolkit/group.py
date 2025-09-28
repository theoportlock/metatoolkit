#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Group - Groups a dataset')
    parser.add_argument('subject', help="Path to the input data file or subject name")
    parser.add_argument('--group_by', nargs='+', help="Columns to group by (+ nargs)")
    parser.add_argument('--func', required=True, help="Aggregation function (e.g., sum, mean)")
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
        # assume subject name: look for results/{subject}.tsv
        return pd.read_csv(Path('results') / f'{path_or_name}.tsv', sep='\t', index_col=0)


def merge_meta(df, meta_path, group_by):
    mdf = pd.read_csv(meta_path, sep='\t', index_col=0)

    if not group_by:
        raise ValueError("--group_by must be specified when using --meta")

    # keep only required grouping columns
    missing = [col for col in group_by if col not in mdf.columns]
    if missing:
        raise ValueError(f"Metadata file missing required columns: {missing}")

    mdf = mdf[group_by]
    df = df.join(mdf, how='inner')
    return df


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
    args = parse_arguments()
    subject = args.subject

    # Determine output filename
    outname = args.output or f"results/{Path(subject).stem}_{args.func}.tsv"
    outpath = Path(outname)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Load subject data
    df = load_data(subject)

    # Merge metadata if specified
    if args.meta:
        df = merge_meta(df, args.meta, args.group_by)

    # Apply grouping
    out = group(df, group_by=args.group_by, func=args.func, axis=args.axis)

    # Print and save
    print(out)
    out.to_csv(outpath, sep='\t')


if __name__ == "__main__":
    main()

