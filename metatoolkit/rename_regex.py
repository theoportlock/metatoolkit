#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Rename dataframe columns or index using regex."
    )
    parser.add_argument("subject", help="Input dataframe file or identifier.")
    parser.add_argument("--match", help="Regex pattern to match.", required=True)
    parser.add_argument("--replace", help="Replacement pattern.", required=True)
    parser.add_argument("--axis", choices=["columns", "index"], default="columns",
                        help="Axis to rename (default: columns).")
    parser.add_argument("-o", "--output", help="Output filename for the renamed dataframe")

    known, unknown = parser.parse_known_args()
    return vars(known), unknown


def save_dataframe(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=df.index.name is not None)


def load_dataframe(subject):
    path = subject if os.path.isfile(subject) else f"results/{subject}.tsv"
    return pd.read_csv(path, sep="\t", index_col=0)


def regex_replace(df, match_pattern, replace_pattern, axis):
    if axis == "index":
        new_labels = [re.sub(match_pattern, replace_pattern, str(label)) for label in df.index]
        df.index = new_labels
    else:
        new_labels = [re.sub(match_pattern, replace_pattern, str(label)) for label in df.columns]
        df.columns = new_labels
    return df


if __name__ == "__main__":
    known_args, _ = parse_arguments()
    subject = known_args.pop("subject")
    subject_name = Path(subject).stem if os.path.isfile(subject) else subject

    df = load_dataframe(subject)

    df = regex_replace(
        df,
        match_pattern=known_args["match"],
        replace_pattern=known_args["replace"],
        axis=known_args["axis"]
    )

    output_filename = known_args.get("output")
    save_dataframe(df, output_filename)
    print(f"Saved regex-renamed dataframe to: {output_filename}")

