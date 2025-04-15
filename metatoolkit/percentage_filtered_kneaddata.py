#!/usr/bin/env python3
import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_file>")
        sys.exit(1)
    
    # Read the input CSV file
    input_file = sys.argv[1]
    df = pd.read_csv(input_file, sep='\t')
    
    # Calculate the total raw reads for each sample
    df['total_raw'] = df['raw pair1'] + df['raw pair2']
    
    # Calculate the total final reads for each sample
    df['total_final'] = (df['final pair1'] + df['final pair2'] +
                         df['final orphan1'] + df['final orphan2'])
    
    # Calculate the number of reads removed by kneaddata
    df['reads_removed'] = df['total_raw'] - df['total_final']
    
    # Calculate the percentage of reads removed relative to the raw reads
    df['percentage_removed'] = (df['reads_removed'] / df['total_raw']) * 100
    
    # Print the results (Sample and percentage removed)
    print(df[['Sample', 'percentage_removed']])
    
if __name__ == "__main__":
    main()

