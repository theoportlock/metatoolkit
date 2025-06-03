#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from skbio.stats.distance import permanova
import matplotlib.pyplot as plt
import pandas as pd

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def PERMANOVA(df, pval=True, full=False):
    np.random.seed(0)
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    result = permanova(DM_dist, df.index)
    if full:
        return result
    if pval:
        return result['p-value']
    else:
        return result['test statistic']
# load and format data
meta = pd.read_csv('results/meta.tsv', sep='\t', index_col=0)

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject): subject = Path(subject).stem
    if tl: plt.tight_layout()
    plt.savefig(f'results/{subject}.svg')
    plt.clf()

dataset_names = sys.argv
if len(sys.argv)<2:
    raise SystemExit('Please add dataset(s) as arguments to this command')
filterdatasets = {i:pd.read_csv(i, sep='\t', index_col=0) for i in sys.argv[1:]}

# Calculate PERMANOVA power
output = pd.Series(index=filterdatasets.keys())
pval=True
for name, tdf in filterdatasets.items():
    #tdf = tdf.loc[tdf.index != 0]
    print(tdf)
    output[name] = PERMANOVA(tdf, pval=pval)
power = -output.apply(np.log).sort_values()

power.plot.barh()
plt.axvline(x=-np.log(0.05), color="black", linestyle="--")
plt.xlabel('Explained Variance (-log2(pval))')
plt.tight_layout()
plt.savefig('results/EV.svg')
