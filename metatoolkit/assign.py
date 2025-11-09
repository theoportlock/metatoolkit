#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign a new column using pandas assign syntax."
    )
    parser.add_argument("input", help="Path to input TSV file")
    parser.add_argument("-o", "--output", default="assigned.tsv", help="Output TSV file name")
    parser.add_argument(
        "-a",
        "--assign",
        required=True,
        help='Expression for new column, e.g. \'pregnant = (status == 1) | (status == 2)\'',
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    # Parse expression like: "pregnant = (status == 1) | (status == 2)"
    if "=" not in args.assign:
        raise ValueError("Assignment must include '=' (e.g. 'newcol = expression')")
    colname, expr = [x.strip() for x in args.assign.split("=", 1)]

    # Use pandas.eval for flexible boolean logic
    df[colname] = pd.eval(expr, engine="python", local_dict={"df": df, **locals(), **globals()}, target=df)

    df.to_csv(args.output, sep="\t", index=False)

if __name__ == "__main__":
    main()

