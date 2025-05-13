#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute percentage of reads removed by KneadData.
"""

import pandas as pd
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate percentage of reads removed by KneadData.")
    parser.add_argument("input", help="Input TSV file with KneadData summary")
    parser.add_argument("output", help="Output TSV file with results")
    
    args = parser.parse_args()
    
    # Read the input TSV file
    df = pd.read_csv(args.input, sep='\t')
    
    # Calculate totals and removal stats
    df['total_raw'] = df['raw pair1'] + df['raw pair2']
    df['total_final'] = df['final pair1'] + df['final pair2'] + df['final orphan1'] + df['final orphan2']
    df['reads_removed'] = df['total_raw'] - df['total_final']
    df['percentage_removed'] = (df['reads_removed'] / df['total_raw']) * 100
    
    # Select and save relevant output
    df[['Sample', 'percentage_removed']].to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()
