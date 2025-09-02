#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skbio.stats.composition import clr, multi_replace
import warnings

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, output_path, index=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def norm(df, axis=0):
    return df.div(df.sum(axis=axis), axis=1-axis)

def standard(df, axis=0):
    if axis == 1:
        return df.T.pipe(standard).T
    return pd.DataFrame(
        StandardScaler().fit_transform(df),
        index=df.index,
        columns=df.columns
    )

def minmax(df, axis=0):
    if axis == 1:
        return df.T.pipe(minmax).T
    return pd.DataFrame(
        MinMaxScaler().fit_transform(df),
        index=df.index,
        columns=df.columns
    )

def log(df, axis=0):
    return df.apply(np.log1p, axis=axis)

def CLR(df, axis=0):
    if axis != 0:
        raise ValueError("CLR is only defined for axis=0 (features in columns).")
    # Mandatory multiplicative replacement before CLR
    tdf = multi_replace(df)
    clr_values = clr(tdf)
    return pd.DataFrame(clr_values, index=df.index, columns=df.columns)

def mult(df, axis=0):
    if axis != 0:
        raise ValueError("Multiplicative replacement only supports axis=0.")
    return pd.DataFrame(multi_replace(df), index=df.index, columns=df.columns)

def hellinger(df, axis=0):
    return np.sqrt(norm(df, axis=axis))

def zscore_rows(df, axis=0):
    if axis == 1:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0), axis=1)

def ALR(df, reference=None):
    if reference is None:
        reference = df.columns[-1]
    if reference not in df.columns:
        raise ValueError(f"Reference column '{reference}' not found in data.")
    return np.log(df.div(df[reference], axis=0).drop(columns=[reference]))

def scale(analysis, df, refcol=None, axis=0):
    available = {
        'norm': lambda x: norm(x, axis=axis),
        'standard': lambda x: standard(x, axis=axis),
        'minmax': lambda x: minmax(x, axis=axis),
        'log': lambda x: log(x, axis=axis),
        'CLR': lambda x: CLR(x, axis=axis),
        'mult': lambda x: mult(x, axis=axis),
        'hellinger': lambda x: hellinger(x, axis=axis),
        'zscore_rows': lambda x: zscore_rows(x, axis=axis),
        'ALR': lambda x: ALR(x, reference=refcol),
    }
    if analysis not in available:
        raise ValueError(f"Unknown analysis method: {analysis}")
    return available[analysis](df)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply a transformation to a dataset.'
    )
    parser.add_argument('analysis', help="Method: norm, standard, minmax, log, CLR, mult, hellinger, zscore_rows, ALR")
    parser.add_argument('subject', help="Input filename or identifier under results/")
    parser.add_argument('--output', '-o', help="Output file path (default: results/{subject}_{analysis}.tsv)")
    parser.add_argument('--refcol', help="Reference column for ALR (default: last column)")
    parser.add_argument('--axis', type=int, choices=[0, 1], default=0,
                        help="Apply transform by columns (0, default) or rows (1) where applicable.")
    return parser.parse_args()

def main():
    args = parse_args()
    subject = args.subject
    analysis = args.analysis
    refcol = args.refcol
    axis = args.axis
    output_file = args.output or f"results/{subject}_{analysis}.tsv"

    df = load(subject)
    output = scale(analysis, df, refcol=refcol, axis=axis)

    print(output)
    save(output, output_file)

if __name__ == "__main__":
    main()

