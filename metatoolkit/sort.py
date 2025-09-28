#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sort - Sorts a dataset")
    parser.add_argument("subject", help="Path to the input data file or subject name")
    parser.add_argument("--axis", type=int, default=0, choices=[0, 1],
                        help="Axis to sort along: 0 = index (rows), 1 = columns")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending (default is descending)")
    parser.add_argument("--df2", help="Optional second dataframe to use as sort order")
    parser.add_argument("--df2_axis", type=int, choices=[0, 1],
                        help="If --df2 is provided, whether to use its index (0) or columns (1)")
    parser.add_argument("-o", "--output", help="Path to save the output file")
    return parser.parse_args()


def load_data(path_or_name):
    path = Path(path_or_name)
    if path.is_file():
        return pd.read_csv(path, sep="\t", index_col=0)
    else:
        return pd.read_csv(Path("results") / f"{path_or_name}.tsv", sep="\t", index_col=0)


def sort_df(df, axis=0, ascending=False, df2=None, df2_axis=None):
    """
    Sort dataframe by its own index/columns or by a second dataframe's index/columns.

    Parameters:
    - df: main dataframe
    - axis: 0 = rows (index), 1 = columns
    - ascending: whether to sort ascending (default False = descending)
    - df2: optional dataframe to use for sort order
    - df2_axis: 0 = index, 1 = columns for df2

    Returns:
    - sorted dataframe
    """
    if df2 is not None:
        if df2_axis is None:
            raise ValueError("--df2_axis must be specified if --df2 is used")

        if df2_axis == 0:  # use df2's index
            order = df2.index
            return df.loc[df.index.intersection(order)].reindex(order, axis=0)
        else:  # use df2's columns
            order = df2.columns
            return df.loc[:, df.columns.intersection(order)].reindex(order, axis=1)

    else:
        if axis == 0:
            return df.sort_index(ascending=ascending, axis=0)
        else:
            return df.sort_index(ascending=ascending, axis=1)


def main():
    args = parse_arguments()
    subject = args.subject

    outname = args.output or f"results/{Path(subject).stem}_sorted.tsv"
    outpath = Path(outname)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Load subject dataframe
    df = load_data(subject)

    # Optional secondary dataframe
    df2 = load_data(args.df2) if args.df2 else None

    # Perform sorting
    out = sort_df(df, axis=args.axis, ascending=args.ascending,
                  df2=df2, df2_axis=args.df2_axis)

    print(out)
    out.to_csv(outpath, sep="\t")


if __name__ == "__main__":
    main()

