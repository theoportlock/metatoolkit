#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rename dataframe columns or index using one or more metadata columns. "
                    "For multiple levels, separate them with commas (e.g. level1,level2)."
    )
    parser.add_argument("subject", help="Input dataframe file or identifier.")
    parser.add_argument("level", help="Metadata column name(s) to use for renaming. "
                                      "For multiple levels, separate by commas.")
    parser.add_argument("--df2", help="Metadata dataframe file or identifier.")
    parser.add_argument("--axis", choices=["columns", "index"], default="index",
                        help="Axis to rename (default: index).")
    parser.add_argument("--output", help="Output filename for the renamed dataframe (without extension).")
    
    known, unknown = parser.parse_known_args()
    return vars(known), unknown


def save_dataframe(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=df.index.name is not None)


def load_dataframe(subject):
    path = subject if os.path.isfile(subject) else f"../results/{subject}.tsv"
    return pd.read_csv(path, sep="\t", index_col=0)


def rename_dataframe(df, meta, levels, axis, join_sep="_"):
    """
    Rename dataframe index or columns using one or more metadata columns.
    If levels is a list, join their values (as strings) with join_sep.
    """
    if isinstance(levels, list):
        # Drop rows where any of the level columns are missing
        meta_clean = meta.dropna(subset=levels)
        mapping = {
            key: join_sep.join(str(meta_clean.loc[key, level]) for level in levels)
            for key in meta_clean.index
        }
    else:
        mapping = meta[levels].dropna().to_dict()

    renamed_df = df.rename(**{axis: mapping})
    return renamed_df if not renamed_df.empty else None


if __name__ == "__main__":
    known_args, _ = parse_arguments()
    subject = known_args.pop("subject")
    subject_name = Path(subject).stem if os.path.isfile(subject) else subject
    
    # Convert the level argument to a list if multiple levels are provided (comma-separated)
    level_arg = known_args["level"]
    if "," in level_arg:
        levels = [lvl.strip() for lvl in level_arg.split(",") if lvl.strip()]
    else:
        levels = level_arg

    df = load_dataframe(subject)
    meta = load_dataframe(known_args.get("df2", "meta"))
    
    output_df = rename_dataframe(df, meta, levels, known_args["axis"])
    
    if output_df is not None:
        output_filename = known_args.get("output")
        if output_filename is None:
            # If multiple levels, join their names for the default filename
            if isinstance(levels, list):
                level_str = "_".join(levels)
            else:
                level_str = levels
            output_filename = f"{subject_name}_{level_str}"
        output_path = f"../results/{output_filename}.tsv"
        save_dataframe(output_df, output_path)
        print(f"Saved renamed dataframe to: {output_path}")
    else:
        print(f"{subject_name} is empty")
