#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Describe - Produces a summary report of taxonomic breakdown')
    parser.add_argument('subject', help='Path to the subject file or subject identifier')
    return parser.parse_args()

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    output_path = f'results/{subject}.tsv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def taxo_summary(df):
    count = df.columns.str.replace(".*\|", "", regex=True).str[0].value_counts().to_frame('all_count')
    output = []
    for sample in df.index:
        samp = df.loc[sample]
        fsamp = samp[samp != 0]
        output.append(fsamp.index.str.replace(".*\|", "", regex=True).str[0].value_counts())
    outdf = pd.concat(output, axis=1).set_axis(df.index, axis=1).T
    odf = pd.concat([
        count,
        outdf.mean().to_frame('mean_count'),
        outdf.std().to_frame('std_count')
    ], axis=1)
    return odf

def main():
    args = parse_args()
    subject = args.subject
    subject_stem = Path(subject).stem if os.path.isfile(subject) else subject

    df = load(subject)
    output = taxo_summary(df)

    print(output.to_string())
    save(output, f'{subject_stem}_taxocount')

if __name__ == '__main__':
    main()
