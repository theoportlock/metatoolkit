#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rename dataframe columns or index using a metadata column."
    )
    parser.add_argument("subject", help="Input dataframe file or identifier.")
    parser.add_argument("level", help="Column in metadata to use for renaming.")
    parser.add_argument("--df2", help="Metadata dataframe file or identifier.")
    parser.add_argument("--axis", choices=["columns", "index"], default="index",
                       help="Axis to rename (default: index).")
    parser.add_argument("--output", help="Output filename for the renamed dataframe")
    
    known, unknown = parser.parse_known_args()
    return vars(known), unknown


def save_dataframe(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=df.index.name is not None)


def load_dataframe(subject):
    path = subject if os.path.isfile(subject) else f"results/{subject}.tsv"
    return pd.read_csv(path, sep="\t", index_col=0)


def rename_dataframe(df, meta, level, axis):
    # Create a mapping dictionary from the metadata, dropping any missing labels
    mapping = meta[level].dropna().to_dict()
    
    # Filter the dataframe to keep only rows/columns that are in the mapping keys
    if axis == "index":
        df = df[df.index.isin(mapping.keys())]
    elif axis == "columns":
        df = df.loc[:, df.columns.isin(mapping.keys())]
    
    # Rename using the mapping
    renamed_df = df.rename(**{axis: mapping})
    return renamed_df if not renamed_df.empty else None


if __name__ == "__main__":
    known_args, _ = parse_arguments()
    subject = known_args.pop("subject")
    subject_name = Path(subject).stem if os.path.isfile(subject) else subject
    
    df = load_dataframe(subject)
    meta = load_dataframe(known_args.get("df2", "meta"))
    
    output_df = rename_dataframe(df, meta, known_args["level"], known_args["axis"])
    
    if output_df is not None:
        # Use the provided output filename or generate a default one
        output_filename = known_args.get("output")
        if output_filename is None:
            output_filename = f"{subject_name}_{known_args['level']}"
        output_path = output_filename
        save_dataframe(output_df, output_path)
        print(f"Saved renamed dataframe to: {output_path}")
    else:
        print(f"{subject_name} is empty")
