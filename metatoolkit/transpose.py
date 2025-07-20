#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Transpose a TSV file, optionally specifying an output file path.'
    )
    parser.add_argument(
        'subject',
        help='Input TSV file path or subject name (if file not found, "results/{subject}.tsv" will be used)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: results/{subject}_T.tsv)'
    )

    args = parser.parse_args()
    subject_path = args.subject

    # Try using subject as a direct file path
    if not os.path.isfile(subject_path):
        subject_path = f'results/{args.subject}.tsv'
        if not os.path.isfile(subject_path):
            raise FileNotFoundError(f'File not found: {args.subject} or results/{args.subject}.tsv')

    # Read and transpose
    df = pd.read_csv(subject_path, sep='\t', index_col=0)
    outdf = df.T

    # Determine output path
    subject_stem = Path(subject_path).stem
    output_path = args.output or f'results/{subject_stem}_T.tsv'

    # Write output
    outdf.to_csv(output_path, sep='\t')
    print(f'Transposed file saved to: {output_path}')

if __name__ == '__main__':
    main()

