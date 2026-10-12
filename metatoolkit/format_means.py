#!/usr/bin/env python3

import argparse
import pandas as pd


def format_mean_std(df, decimals=1):
    """
    Merge *_mean and *_std columns into:
        value (std)

    Example:
        systolic_bp_mean + systolic_bp_std
        ->
        systolic_bp = "120.2 (13.0)"
    """

    output = df.copy()

    mean_cols = [c for c in df.columns if c.endswith("_mean")]

    for mean_col in mean_cols:
        base = mean_col[:-5]  # remove "_mean"
        std_col = f"{base}_std"

        if std_col not in df.columns:
            continue

        merged_col = base

        output[merged_col] = (
            df[mean_col].round(decimals).astype(str)
            + " ("
            + df[std_col].round(decimals).astype(str)
            + ")"
        )

        output.drop(columns=[mean_col, std_col], inplace=True)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Merge mean/std columns into publication format."
    )

    parser.add_argument(
        "input",
        help="Input TSV/CSV file"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file"
    )

    parser.add_argument(
        "--sep",
        default="\t",
        help="Input/output separator (default: tab)"
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Number of decimal places (default: 1)"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)

    formatted = format_mean_std(df, decimals=args.decimals)

    formatted.to_csv(
        args.output,
        sep=args.sep,
        index=False
    )


if __name__ == "__main__":
    main()
