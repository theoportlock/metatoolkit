#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Drop rows or columns from a TSV using pandas drop().")
    parser.add_argument("--infile", required=True, help="Input TSV file")
    parser.add_argument("--outfile", required=True, help="Output TSV file")
    parser.add_argument("--labels", required=True, help="Comma-separated list of row/column labels to drop")
    parser.add_argument("--axis", type=int, choices=[0, 1], required=True,
                        help="0 to drop rows, 1 to drop columns")
    parser.add_argument("--index-col", type=int, default=0,
                        help="Column to use as index (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the DataFrame
    df = pd.read_csv(args.infile, sep="\t", index_col=args.index_col)

    # Parse comma-separated labels
    labels = [label.strip() for label in args.labels.split(",")]

    # Perform drop
    df_dropped = df.drop(labels=labels, axis=args.axis, errors='ignore')

    # Save the result
    df_dropped.to_csv(args.outfile, sep="\t")
    print(f"Dropped {len(labels)} item(s) along axis {args.axis} and saved to {args.outfile}")


if __name__ == "__main__":
    main()

