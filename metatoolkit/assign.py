#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign a new column using pandas eval."
    )
    parser.add_argument("input", help="Path to input TSV file")
    parser.add_argument("-o", "--output", default="assigned.tsv")
    parser.add_argument(
        "-a", "--assign",
        required=True,
        help="Assignment expression, e.g. 'newcol = colA + colB'"
    )
    parser.add_argument(
        "--engine",
        choices=["python", "numexpr"],
        default="python",
        help="Evaluation engine (default: python)"
    )
    parser.add_argument(
        "--mode",
        choices=["dataframe", "global"],
        default="dataframe",
        help="Evaluation mode: dataframe=use df.eval, global=use pd.eval"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    if "=" not in args.assign:
        raise ValueError("Assignment must include '='")

    colname, expr = [x.strip() for x in args.assign.split("=", 1)]

    if args.mode == "dataframe":
        # filename + "_" + model
        df[colname] = df.eval(expr, engine=args.engine)
    else:
        # df["filename"] + "_" + df["model"]
        df[colname] = pd.eval(
            expr,
            engine=args.engine,
            local_dict={"df": df},
            target=df
        )
    df.to_csv(args.output, sep="\t", index=False)
    print(df.head())

if __name__ == "__main__":
    main()

