#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from itertools import permutations

parser = argparse.ArgumentParser(description='''
Corr - Produces a report of the significant correlations between data
''')
parser.add_argument('subject', nargs='+')
parser.add_argument('-m', '--mult', action='store_true')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
mult = known.get("mult") if known.get('mult') else False

def corrpair(df1, df2, FDR=True, min_unique=0):
    df1 = df1.loc[:, df1.nunique() > min_unique]
    df2 = df2.loc[:, df2.nunique() > min_unique]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    pvaldf.fillna(1, inplace=True)
    if FDR:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    return cordf, pvaldf

def corr(df):
    combs = list(permutations(df.columns.unique(), 2))
    outdf = pd.DataFrame(index=pd.MultiIndex.from_tuples(combs), columns=['cor','pval'])
    for comb in combs:
        tdf = pd.concat([df[comb[0]], df[comb[1]]], axis=1).dropna()
        cor, pval = spearmanr(tdf[comb[0]], tdf[comb[1]])
        outdf.loc[comb, 'cor'] = cor
        outdf.loc[comb, 'pval'] = pval
    outdf['qval'] = fdrcorrection(outdf.pval)[1]
    outdf.index.set_names(['source', 'target'], inplace=True)
    return outdf

if len(subject) == 1:
    df = f.load(subject[0])
    output = corr(df)
    print(output)
    f.save(output, subject[0]+'corr')
elif len(subject) == 2:
    df1 = f.load(subject[0])
    df2 = f.load(subject[1])
    cor,pval = corrpair(df1, df2)
    print(cor)
    f.save(cor, subject[0] + subject[1] +'corr')
    f.save(pval, subject[0] + subject[1] +'corrpval')
else:
    print('invalid number of arguments')

