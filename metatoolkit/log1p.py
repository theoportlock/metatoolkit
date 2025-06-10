#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import os

def load(subject):
    return pd.read_csv(subject, sep='\t', index_col=0)

def save(df, subject, index=True):
    if not subject.endswith('.tsv'):
        subject = f'results/{subject}.tsv'
    output_path = subject
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def log1p(df):
    out = df.copy()
    out = out.apply(np.log1p)
    return out 

def parse_args():
    parser = argparse.ArgumentParser(description='Performs a log1p adjustment')
    parser.add_argument('--profile', required=True, help='Input profile file')
    parser.add_argument('-o', '--outfile', type=str, help='Output file name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the original DataFrame
    df = load(args.profile)

    # Filter the DataFrame using the provided options
    output = log1p(df)
    
    # Save results
    save(output, args.outfile)

if __name__ == "__main__":
    main()
