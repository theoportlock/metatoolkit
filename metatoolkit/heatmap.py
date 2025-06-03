#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import os

def heatmap(df, sig=None, ax=None, center=0, **kwargs):
    pd.set_option("use_inf_as_na", True)
    if ax is None: fig, ax= plt.subplots()
    g = sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=center,
        yticklabels=True,
        xticklabels=True,
        annot=sig.replace({True:'*',False:''}) if sig is not None else None,
        fmt='',
    )
    plt.setp(g.get_xticklabels(), rotation=40, ha="right")
    return g
parser = argparse.ArgumentParser(description='''
Heatmap - Produces a heatmap of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = pd.read_csv(subject, index_col=0, sep='\t')

heatmap(df)
plt.savefig(f'results/{subject}heatmap.svg')
