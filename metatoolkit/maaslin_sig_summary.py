#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path


def change_summary(df, change="coef", sig="qval", pval=0.25):
    """Generate a summary of significant changes based on thresholds."""
    total_rows = df.shape[0]

    sig_mask = df[sig] < pval

    sig_changed = sig_mask.sum()
    sig_up = (sig_mask & (df[change] > 0)).sum()
    sig_down = (sig_mask & (df[change] < 0)).sum()

    return pd.Series({
        "n_tested": total_rows,
        "n_sig": sig_changed,
        "n_sig_up": sig_up,
        "n_sig_down": sig_down,
        "pct_sig": 100 * sig_changed / total_rows if total_rows else 0,
        "pct_sig_up": 100 * sig_up / total_rows if total_rows else 0,
        "pct_sig_down": 100 * sig_down / total_rows if total_rows else 0,
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise MaAsLin results with optional grouping by metadata/model."
    )
    parser.add_argument(
        "subject",
        help="Input MaAsLin results TSV."
    )
    parser.add_argument(
        "-p", "--pval",
        type=float,
        default=0.25,
        help="P-value or q-value threshold [default: 0.25]."
    )
    parser.add_argument(
        "-c", "--change",
        type=str,
        default="coef",
        help="Column name for change effect [default: coef]."
    )
    parser.add_argument(
        "-s", "--sig",
        type=str,
        default="qval_joint",
        help="Column name for significance [default: qval_joint]."
    )
    parser.add_argument(
        "-g", "--group-cols",
        nargs="+",
        default=None,
        help=(
            "Column(s) to group by before summarising "
            "(e.g. metadata model). If omitted, summarises entire table."
        )
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path for summary TSV."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    subject_path = Path(args.subject)
    if not subject_path.exists():
        raise FileNotFoundError(f"Input file not found: {subject_path}")

    df = pd.read_csv(subject_path, sep="\t")

    # Validate required columns
    required = {args.change, args.sig}
    if args.group_cols:
        required |= set(args.group_cols)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if args.group_cols:
        summary = (
            df
            .groupby(args.group_cols, dropna=False)
            .apply(
                change_summary,
                change=args.change,
                sig=args.sig,
                pval=args.pval
            )
            .reset_index()
        )
    else:
        summary = change_summary(
            df,
            change=args.change,
            sig=args.sig,
            pval=args.pval
        ).to_frame().T

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"\nSummary written to: {output_path}")


if __name__ == "__main__":
    main()

