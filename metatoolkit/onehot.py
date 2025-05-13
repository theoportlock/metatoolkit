#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from pathlib import Path
import os

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, output_path, index=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def onehot(df):
    return pd.get_dummies(df, prefix_sep='.', dtype=bool)

def parse_args():
    parser = argparse.ArgumentParser(description='One-hot encode columns of a dataset.')
    parser.add_argument('subject', type=str, help='Input data file or subject name')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load input
    subject = args.subject
    df = load(subject)

    df_encoded = onehot(df)

    print(df_encoded)

    # Save to output file
    save(df_encoded, args.output)

if __name__ == '__main__':
    main()

