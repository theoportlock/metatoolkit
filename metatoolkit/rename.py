#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rename dataframe columns or index using a metadata column or index."
    )
    parser.add_argument("subject", help="Input dataframe file or identifier.")
    parser.add_argument("level", help="Metadata column name or index name to use for renaming.")
    parser.add_argument("--df2", default="meta",
                        help="Metadata dataframe file or identifier (default: meta).")
    parser.add_argument("--axis", choices=["columns", "index"], default="index",
                        help="Axis to rename (default: index).")
    parser.add_argument("--output", help="Output filename for the renamed dataframe")

    known, unknown = parser.parse_known_args()
    return vars(known), unknown


def load_dataframe(subject):
    path = subject if os.path.isfile(subject) else f"results/{subject}.tsv"
    return pd.read_csv(path, sep="\t", index_col=0)


def save_dataframe(df, output_path):
    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=True)


def get_mapping_from_metadata(meta, level):
    """
    Resolve `level` to either a metadata column or the metadata index
    and return a mapping dict: {old_id -> new_id}.
    """
    if level in meta.columns:
        series = meta[level]
    elif meta.index.name == level:
        series = pd.Series(meta.index, index=meta.index)
    else:
        raise KeyError(
            f"'{level}' not found in metadata columns or index.\n"
            f"Available columns: {list(meta.columns)}\n"
            f"Index name: {meta.index.name}"
        )

    if series.isna().all():
        raise ValueError(f"Metadata field '{level}' contains only NA values.")

    return series.dropna().to_dict()


def rename_dataframe(df, meta, level, axis):
    mapping = get_mapping_from_metadata(meta, level)

    # Determine keys on the selected axis
    keys = df.index if axis == "index" else df.columns
    overlap = keys.intersection(mapping.keys())

    if overlap.empty:
        raise ValueError(
            f"No matching IDs between dataframe {axis} and metadata.\n"
            f"Example dataframe IDs: {list(keys[:5])}\n"
            f"Example metadata IDs: {list(mapping.keys())[:5]}"
        )

    # Subset first, then rename
    if axis == "index":
        df = df.loc[overlap]
    else:
        df = df.loc[:, overlap]

    renamed_df = df.rename(**{axis: mapping})

    return renamed_df


if __name__ == "__main__":
    known_args, _ = parse_arguments()

    subject = known_args.pop("subject")
    subject_name = Path(subject).stem if os.path.isfile(subject) else subject

    df = load_dataframe(subject)
    meta = load_dataframe(known_args["df2"])

    output_df = rename_dataframe(
        df=df,
        meta=meta,
        level=known_args["level"],
        axis=known_args["axis"],
    )

    output_filename = known_args.get("output")
    if output_filename is None:
        output_filename = f"{subject_name}_{known_args['level']}.tsv"

    save_dataframe(output_df, output_filename)
    print(f"Saved renamed dataframe to: {output_filename}")

