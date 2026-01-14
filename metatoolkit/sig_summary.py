#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def change_summary(df, change='coef', sig='qval', pval=0.25):
    """Generate a summary of significant changes based on thresholds."""
    total_rows = df.shape[0]

    sig_changed_count = df[sig].lt(pval).sum()
    changed = f"sig changed = {sig_changed_count}/{total_rows} ({round(sig_changed_count / total_rows * 100)}%)"

    sig_increased_count = df.loc[(df[sig] < pval) & (df[change] > 0), sig].lt(pval).sum()
    increased = f"sig up = {sig_increased_count}/{total_rows} ({round(sig_increased_count / total_rows * 100)}%)"

    sig_decreased_count = df.loc[(df[sig] < pval) & (df[change] < 0), sig].lt(pval).sum()
    decreased = f"sig down = {sig_decreased_count}/{total_rows} ({round(sig_decreased_count / total_rows * 100)}%)"

    return pd.Series([changed, increased, decreased], name="summary")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Produces a summary report of analysis based on change and significance thresholds."
    )
    parser.add_argument("subject", help="Input dataframe file (TSV).")
    parser.add_argument(
        "-p", "--pval", type=float, default=0.25,
        help="P-value or q-value threshold [default: 0.25]."
    )
    parser.add_argument(
        "-c", "--change", type=str, default="coef",
        help="Column name for change effect [default: coef]."
    )
    parser.add_argument(
        "-s", "--sig", type=str, default="qval",
        help="Column name for significance [default: qval]."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output file path for summary TSV."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    subject_path = Path(args.subject)
    if not subject_path.exists():
        raise FileNotFoundError(f"Input file not found: {subject_path}")

    df = pd.read_csv(subject_path, index_col=0, sep="\t")
    output = change_summary(df, change=args.change, pval=args.pval, sig=args.sig)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(output.to_string())
    output.to_csv(output_path, sep="\t", header=False)
    print(f"\n✅ Summary written to: {output_path}")


if __name__ == "__main__":
    main()

