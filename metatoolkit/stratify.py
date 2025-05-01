#!/opt/anaconda/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratify - Stratifies a dataframe according to a column of another dataframe (commonly metadata)."
    )
    parser.add_argument('subject', help='Subject file or name (without extension)')
    parser.add_argument('level', help='Metadata column to stratify by')
    parser.add_argument('--df2', help='Optional second dataframe (default: meta)')
    parser.add_argument('-o','--output', help='Output file path (default: ../results/{subject}{level}.tsv)')
    return parser.parse_args()


def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)


def save(df, output_path, index=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)


def stratify(df, meta, level):
    metadf = df.join(meta[level].dropna(), how='inner')
    metadf = metadf.reset_index(drop=True).set_index(level).sort_index()
    if metadf.empty:
        return None
    return metadf


def main():
    args = parse_args()

    subject_path = args.subject
    if os.path.isfile(subject_path):
        subject_name = Path(subject_path).stem
    else:
        subject_name = subject_path

    df = load(subject_path)
    meta = load(args.df2) if args.df2 else load('meta')
    level = args.level

    output_df = stratify(df, meta, level)

    if output_df is not None:
        if args.output:
            output_path = args.output
        else:
            output_path = f'../results/{subject_name}{level}.tsv'

        save(output_df, output_path)
        print(f"Saved stratified dataframe to {output_path}")
    else:
        print(f"{subject_name} is empty after stratification.")


if __name__ == "__main__":
    main()
