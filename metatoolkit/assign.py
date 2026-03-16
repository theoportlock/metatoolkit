#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign new columns to a TSV using expressions."
    )
    parser.add_argument(
        "input",
        help="Path to input TSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default="assigned.tsv",
        help="Path to output TSV file (default: assigned.tsv)"
    )
    parser.add_argument(
        "-a", "--assign",
        required=True,
        action="append",
        help='Assignment expression (can be repeated), e.g. -a "newcol = colA + colB"'
    )
    parser.add_argument(
        "--mode",
        choices=["dataframe", "global"],
        default="dataframe",
        help=(
            "Evaluation mode:\n"
            "  dataframe → pandas df.eval (column names only)\n"
            "  global    → full Python eval (df available)\n"
        )
    )
    return parser.parse_args()

def apply_assignment(df, assign_expr, mode):
    if "=" not in assign_expr:
        sys.exit(f"ERROR: assignment must contain '=' → {assign_expr}")

    colname, expr = [x.strip() for x in assign_expr.split("=", 1)]

    if mode == "dataframe":
        df[colname] = df.eval(expr)
    else:
        SAFE_BUILTINS = {
            "str": str,
            "int": int,
            "float": float,
            "len": len,
        }

        result = eval(
            expr,
            {"__builtins__": SAFE_BUILTINS},
            {"df": df}
        )

        df[colname] = result

def main():
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t")

    try:
        for assign_expr in args.assign:
            apply_assignment(df, assign_expr, args.mode)

    except Exception as e:
        sys.exit(f"ERROR evaluating expression:\n{e}")

    df.to_csv(args.output, sep="\t", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
