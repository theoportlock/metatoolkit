#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'../results/{subject}.tsv', sep='\t', index_col=0)

def abund(df, order=None, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    tdf = df.copy()
    if tdf.columns.shape[0] > 20:
        tdf['others'] = tdf[tdf.mean().sort_values(ascending=False).iloc[21:].index].sum(axis=1)
    tdf = norm(tdf.loc[:, tdf.mean().sort_values(ascending=False).iloc[:20].index])
    if order:
        tdf = tdf.loc[:, tdf.mean().sort_values(ascending=True).index]
    tdf.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def savefig(subject, tl=False, show=False):
    if os.path.isfile(subject): subject = Path(subject).stem
    if tl: plt.tight_layout()
    plt.savefig(f'../results/{subject}.svg')
    plt.savefig(f'../results/{subject}.pdf')
    plt.savefig(f'../results/{subject}.jpg', dpi=500)
    if show: plt.show()
    subprocess.call(f'zathura ../results/{subject}.pdf &', shell=True)
    plt.clf()

parser = argparse.ArgumentParser(description='''
Abund - Produces an Abundance/Relative abundance plot of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = load(subject)

abund(df)
savefig(f'{subject}abund')
