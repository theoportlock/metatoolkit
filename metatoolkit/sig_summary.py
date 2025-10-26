#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def change_summary(df, change='coef', sig='qval', pval=0.25):
    """Summarize number of significant changes."""
    total_rows = df.shape[0]
    sig_changed_count = df[sig].lt(pval).sum()
    changed = f"sig changed = {sig_changed_count}/{total_rows} ({round(sig_changed_count / total_rows * 100)}%)"

    sig_increased_count = df.loc[(df[sig] < pval) & (df[change] > 0)].shape[0]
    increased = f"sig up = {sig_increased_count}/{total_rows} ({round(sig_increased_count / total_rows * 100)}%)"

    sig_decreased_count = df.loc[(df[sig] < pval) & (df[change] < 0)].shape[0]
    decreased = f"sig down = {sig_decreased_count}/{total_rows} ({round(sig_decreased_count / total_rows * 100)}%)"

    return pd.Series([changed, increased, decreased], index=["changed", "increased", "decreased"])


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize significant changes from a results table.")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("qval_joined.tsv"),
        help="Input TSV file (default: qval_joined.tsv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("summary.tsv"),
        help="Output TSV file (default: summary.tsv)"
    )
    parser.add_argument(
        "-p", "--pval",
        type=float,
        default=0.25,
        help="Significance threshold (default: 0.25)"
    )
    parser.add_argument(
        "-c", "--change",
        type=str,
        default="coef",
        help="Column name for change values (default: coef)"
    )
    parser.add_argument(
        "-s", "--sig",
        type=str,
        default="qval",
        help="Column name for significance values (default: qval)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, index_col=0, sep="\t")
    output = change_summary(df, change=args.change, sig=args.sig, pval=args.pval)

    print(output.to_string(index=False))
    output.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()

