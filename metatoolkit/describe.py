#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
parser.add_argument('subject')
parser.add_argument('-p', '--pval', type=float)
parser.add_argument('-c', '--change')
parser.add_argument('-s', '--sig')
parser.add_argument('-r', '--corr', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def save(df, subject, index=True):
    output_path = f'results/{subject}.tsv' 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=index)

def Describe(df, datatype=None, **kwargs):
    available = {'change_summary':change_summary,
                 'corr_summary':corr_summary,
                 'mbiome_summary':mbiome_summary}
    output = available.get(datatype)(df, kwargs)
    return output

def change_summary(df, kwargs):
    pval = kwargs.get('pval') if kwargs.get('pval') else 0.05
    change = kwargs.get('change')
    sig = kwargs.get('sig')
    print(pval, change, sig)
    assert change and sig
    changed = 'sig changed = ' +\
        str(df[sig].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df[sig].lt(pval).sum()/df.shape[0] * 100)) + '%)'
    increased = 'sig increased = ' +\
        str(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df[sig].lt(pval)) & (df[change].gt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    decreased = 'sig decreased = ' +\
        str(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df[sig].lt(pval)) & (df[change].lt(0)),sig].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    outdf = pd.Series([changed, increased, decreased])
    return outdf

def corr_summary(df, kwargs):
    changed = 'sig correlated = ' +\
        str(df['sig'].lt(pval).sum()) + '/' + str(df.shape[0]) + ' (' + str(round(df['sig'].lt(pval).sum()/df.shape[0] * 100)) + '%)'
    increased = 'sig positively correlated = ' +\
        str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].gt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    decreased = 'sig negatively correlated = ' +\
        str(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()) +\
        '/' +\
        str(df.shape[0]) +\
        ' (' +\
        str(round(df.loc[(df['sig'].lt(pval)) & (df[df.columns[df.columns.str.contains('rho')]].lt(0).iloc[:,0]), 'sig'].lt(pval).sum()/df.shape[0] * 100)) +\
        '%)'
    summary = pd.Series([changed, increased, decreased])
    return summary

def mbiome_summary(df, kwargs):
    return outdf

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = load(subject)

df = load(subject)
output = Describe(df, **known)
print(output.to_string())
save(output, f'{subject}Describe')
