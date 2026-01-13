#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute read depth and percentage of reads filtered by KneadData.
"""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize KneadData output with read depth and percentage filtered."
    )
    parser.add_argument(
        "input",
        help="Input kneaddata_summary.tsv file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output TSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read KneadData summary
    df = pd.read_csv(args.input, sep="\t", index_col=0)

    # -----------------------------
    # Read depth (HQ reads, NOT millions)
    # -----------------------------
    final_cols = df.columns[df.columns.str.contains("final")]
    df["hq_read_depth"] = df[final_cols].sum(axis=1)

    # -----------------------------
    # Percentage filtered
    # -----------------------------
    df["total_raw"] = df["raw pair1"] + df["raw pair2"]
    df["total_final"] = (
        df["final pair1"]
        + df["final pair2"]
        + df["final orphan1"]
        + df["final orphan2"]
    )

    df["reads_removed"] = df["total_raw"] - df["total_final"]
    df["percentage_filtered"] = (df["reads_removed"] / df["total_raw"]) * 100

    # -----------------------------
    # Output
    # -----------------------------
    out = df[[
        "hq_read_depth",
        "percentage_filtered"
    ]].reset_index().rename(columns={"index": "Sample"})

    out.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()

