#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''
def fc(df):
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection
    df = df.loc[:, (df.groupby(level=[0, 1]).nunique() > 3).all()]
    logfcdf = df.div(
        df.xs('BASELINE', level=0).droplevel(0),
        level=2
    ).apply(np.log2)
    #logfcdf.replace([np.inf, -np.inf], 0, inplace=True)
    #logfcdf.dropna(inplace=True, axis=1)
    #logfcdf.fillna(0, inplace=True)
    logfcdf = logfcdf.groupby(level=[0, 1]).median().T
    logfcdf = logfcdf.T.drop('BASELINE', level=0).T
    def appl(df, baseline):
        result = pd.Series(dtype='float64')
        baselinevar = df.reset_index()['ARM'].unique()[0]
        for i in df.columns:
            result[i] = mannwhitneyu(baseline.loc[baselinevar,i],df[i])[1]
        return result
    baseline = df.xs('BASELINE', level=0)
    notbaseline = df.drop('BASELINE', level=0)
    pvaldf = notbaseline.groupby(level=[0, 1]).apply(appl, (baseline)).T
    pvaldf.replace([np.inf, -np.inf], np.nan, inplace=True)
    pvaldf.dropna(inplace=True)
    '''
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[0].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    '''
    return logfcdf, pvaldf

def corr(df1, df2):
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import fdrcorrection
    df1 = df1.loc[:,df1.nunique() > 5]
    df2 = df2.loc[:,df2.nunique() > 5]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df.values)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[0].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return cordf, qvaldf

def heatmap(df):
    import seaborn as sns
    sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )

def clustermap(df):
    import seaborn as sns
    sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )
