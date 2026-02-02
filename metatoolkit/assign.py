#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign a new column to a TSV using an expression."
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
        help="Assignment expression, e.g. \"newcol = colA + colB\""
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

def main():
    args = parse_args()

    # Read input
    df = pd.read_csv(args.input, sep="\t")

    # Parse assignment
    if "=" not in args.assign:
        sys.exit("ERROR: assignment must contain '='")

    colname, expr = [x.strip() for x in args.assign.split("=", 1)]

    try:
        if args.mode == "dataframe":
            # Example: newcol = colA + colB
            df[colname] = df.eval(expr)
        else:
            # Example: newcol = 'S' + df.subjectID.astype(str)
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

    except Exception as e:
        sys.exit(f"ERROR evaluating expression:\n{e}")

    # Write output
    df.to_csv(args.output, sep="\t", index=False)
    print(df.head())

if __name__ == "__main__":
    main()

