#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd
from pandas.api.types import is_object_dtype
from pandas.api.types import CategoricalDtype


def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep="\t", index_col=0)
    return pd.read_csv(f"results/{subject}.tsv", sep="\t", index_col=0)


def save(df, output_path, index=True):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=index)


def sanitize_headers(columns):
    return [re.sub(r"[^0-9a-zA-Z_\-]", "_", col) for col in columns]


def onehot(df, include_cols=None, exclude_cols=None, prefix_sep=".", dtype="int"):
    if include_cols:
        cols_to_encode = [c for c in include_cols if c in df.columns]
    else:
        if exclude_cols:
            candidates = [c for c in df.columns if c not in exclude_cols]
        else:
            candidates = df.columns.tolist()

        cols_to_encode = [
            c
            for c in candidates
            if is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype)
        ]

    if not cols_to_encode:
        return df.copy()

    df_to_encode = df[cols_to_encode]
    df_remaining = df.drop(columns=cols_to_encode)

    df_encoded = pd.get_dummies(df_to_encode, prefix_sep=prefix_sep)

    # Force proper dtype now
    if dtype == "bool":
        df_encoded = df_encoded.astype(bool)
    elif dtype == "float":
        df_encoded = df_encoded.astype(float)
    else:
        df_encoded = df_encoded.astype(int)

    return pd.concat([df_remaining, df_encoded], axis=1)



def parse_args():
    parser = argparse.ArgumentParser(description="One-hot encode categorical columns of a dataset.")
    parser.add_argument("subject", type=str, help="Input data file or subject name")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--include-cols",
        type=str,
        help="Comma-separated list of columns to include in one-hot encoding",
    )
    parser.add_argument(
        "--exclude-cols",
        type=str,
        help="Comma-separated list of columns to exclude from one-hot encoding",
    )
    parser.add_argument(
        "--prefix-sep", type=str, default=".", help='Prefix separator used in one-hot encoding (default: ".")'
    )
    parser.add_argument(
        "--sanitize-headers",
        action="store_true",
        help="Sanitize column headers in output to letters, digits, dashes and underscores only",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=None,
        help="Drop one-hot columns whose fraction of TRUE (1) values is below this threshold (0–1)",
    )
    parser.add_argument(
        "--drop-onehot-values",
        type=str,
        default=None,
        help="Comma-separated list of level names to drop from one-hot columns "
        "(matches the string after the prefix separator)",
    )
    parser.add_argument(
        "--dtype", type=str, default="int", choices=["int", "float", "bool"],
        help="Data type of one-hot encoded columns (default: int)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load(args.subject)
    include_cols = [c.strip() for c in args.include_cols.split(",")] if args.include_cols else None
    exclude_cols = [c.strip() for c in args.exclude_cols.split(",")] if args.exclude_cols else None

    df_encoded = onehot(
        df,
        include_cols=include_cols,
        exclude_cols=exclude_cols,
        prefix_sep=args.prefix_sep,
        dtype=args.dtype,
    )

    # --- Drop low-prevalence one-hot columns
    if args.min_prevalence is not None:
        mask = df_encoded.mean(axis=0) >= args.min_prevalence
        df_encoded = df_encoded.loc[:, mask]

    # --- Drop columns where the level name after the separator matches given strings
    if args.drop_onehot_values:
        drop_values = [v.strip() for v in args.drop_onehot_values.split(",")]
        sep = args.prefix_sep
        keep_cols = []
        for c in df_encoded.columns:
            if sep in c:
                level = c.split(sep)[-1]
                if level in drop_values:
                    continue
            keep_cols.append(c)
        df_encoded = df_encoded[keep_cols]

    if args.sanitize_headers:
        df_encoded.columns = sanitize_headers(df_encoded.columns)

    save(df_encoded, args.output)


if __name__ == "__main__":
    main()

