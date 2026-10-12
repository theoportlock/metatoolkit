#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Python/Pandas expression across an entire DataFrame."
    )
    parser.add_argument("input", help="Path to input TSV file")
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output TSV file"
    )
    parser.add_argument(
        "-e", "--expr",
        required=True,
        help='Expression to evaluate, using "df" as the variable. e.g., "df < 0" or "df.fillna(0)"'
    )
    parser.add_argument(
        "--index-col",
        default=0,
        type=int,
        help="Index column position to preserve row names (default: 0). Set to None if no index."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Handle 'None' string for index_col
    index_col = None if str(args.index_col).lower() == 'none' else args.index_col

    # Load the dataframe
    df = pd.read_csv(args.input, sep="\t", index_col=index_col)

    try:
        # Evaluate the expression
        result = eval(args.expr, {"__builtins__": {}}, {"df": df})
    except Exception as e:
        sys.exit(f"ERROR evaluating expression '{args.expr}':\n{e}")

    # Format check before saving
    if isinstance(result, pd.Series):
        result = result.to_frame()
    elif not isinstance(result, pd.DataFrame):
        sys.exit(f"ERROR: Expression did not return a DataFrame or Series. Returned type: {type(result)}")

    # Ensure output base directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    result.to_csv(args.output, sep="\t")
    print(f"Success! Evaluated '{args.expr}' -> {args.output}")

if __name__ == "__main__":
    main()
