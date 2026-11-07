#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import json
from pathlib import Path
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rename dataframe columns or index using regex or multiple mappings."
    )
    parser.add_argument(
        "subject",
        help="Input dataframe file or identifier (TSV format).",
    )
    parser.add_argument(
        "--match",
        help="Regex pattern to match (ignored if --map is provided).",
    )
    parser.add_argument(
        "--replace",
        help="Replacement pattern (used with --match). Can be a string or a Python lambda.",
    )
    parser.add_argument(
        "--map",
        help=(
            "JSON or comma-separated key:value pairs for multiple replacements, "
            'e.g. \'{"old1":"new1", "old2":"new2"}\'.'
        ),
    )
    parser.add_argument(
        "--axis",
        choices=["columns", "index"],
        default="columns",
        help="Axis to rename (default: columns).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename for the renamed dataframe",
        required=True,
    )

    return parser.parse_args()


def save_dataframe(df, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df.to_csv(
        output_path,
        sep="\t",
        index=True,
        index_label=df.index.name,
    )


def load_dataframe(subject):
    path = subject if os.path.isfile(subject) else f"results/{subject}.tsv"

    return pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )


def apply_mapping(df, mapping, axis):
    """Apply a dict-based rename while preserving index/column names."""

    index_name = df.index.name
    column_names = df.columns.names

    if axis == "index":
        df.rename(index=mapping, inplace=True)
        df.index.name = index_name
    else:
        df.rename(columns=mapping, inplace=True)
        df.columns.names = column_names

    return df


def regex_replace(df, match_pattern, replace_pattern, axis):
    """Perform regex-based renaming (supports lambdas)."""

    # preserve names
    index_name = df.index.name
    column_names = df.columns.names

    # allow lambda replacements
    if (
        replace_pattern
        and isinstance(replace_pattern, str)
        and replace_pattern.strip().startswith("lambda")
    ):
        repl = eval(replace_pattern)
    else:
        repl = replace_pattern

    if axis == "index":
        df.index = [
            re.sub(match_pattern, repl, str(label))
            for label in df.index
        ]
        df.index.name = index_name

    else:
        df.columns = [
            re.sub(match_pattern, repl, str(label))
            for label in df.columns
        ]
        df.columns.names = column_names

    return df


if __name__ == "__main__":
    args = parse_arguments()

    df = load_dataframe(args.subject)

    # If --map is provided, handle multiple replacements at once
    if args.map:
        try:
            mapping = json.loads(args.map)

        except json.JSONDecodeError:
            # fallback for "a:b,c:d" syntax
            mapping = dict(
                pair.split(":")
                for pair in args.map.split(",")
            )

        df = apply_mapping(df, mapping, args.axis)

    # Otherwise, fall back to regex mode
    elif args.match and args.replace is not None:
        df = regex_replace(
            df,
            args.match,
            args.replace,
            args.axis,
        )

    else:
        raise ValueError(
            "You must specify either --map or both --match and --replace."
        )

    save_dataframe(df, args.output)

    print(f"✓ Saved renamed DataFrame → {args.output}")
