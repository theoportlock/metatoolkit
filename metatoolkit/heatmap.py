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

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject): subject = Path(subject).stem
    if tl: plt.tight_layout()
    plt.savefig(f'results/{subject}.svg')
    plt.clf()

parser = argparse.ArgumentParser(description='''
Heatmap - Produces a heatmap of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")

if os.path.isfile(subject): subject = Path(subject).stem
df = load(subject)

heatmap(df)
savefig(f'{subject}_heatmap')
