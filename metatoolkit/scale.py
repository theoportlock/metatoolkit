#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    output_path = f'results/{subject}.tsv' 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def zscore(df, axis=0):
    return df.apply(zscore, axis=axis)

def standard(df):
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def minmax(df):
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def log(df):
    return df.apply(np.log1p)

def CLR(df):
    return df.T.apply(clr).T

def mult(df):
    return pd.DataFrame(mul(df.T).T, index=df.index, columns=df.columns)

def scale(analysis, df):
    available={
        'norm':norm,
        'standard':standard,
        'minmax':minmax,
        'log':log,
        'CLR':CLR,
        'mult':mult,
        }
    output = available.get(analysis)(df)
    return output

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

subject = known.get("subject")
analysis = known.get('analysis')
df = load(subject)

output = scale(analysis, df)
print(output)
save(output, subject+analysis)
