#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import seaborn as sns

def spindle(df, meta=None, ax=None, palette=None, **kwargs):
    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    if ax is None: fig, ax= plt.subplots()
    x=df.columns[0]
    y=df.columns[1]
    centers = df.groupby(df.index).mean()
    centers.columns=['nPC1','nPC2']
    j = df.join(centers)
    j['colours'] = palette
    i = j.reset_index().index[0]
    for i in j.reset_index().index:
        ax.plot(
            j[[x,'nPC1']].iloc[i],
            j[[y,'nPC2']].iloc[i],
            linewidth = 0.5,
            color = j['colours'].iloc[i],
            zorder=1,
            alpha=0.3
        )
        ax.scatter(j[x].iloc[i], j[y].iloc[i], color = j['colours'].iloc[i], s=1)
    for i in centers.index:
        ax.text(centers.loc[i,'nPC1']+0.002,centers.loc[i,'nPC2']+0.002, s=i, zorder=3)
    ax.scatter(centers.nPC1, centers.nPC2, c='black', zorder=2, s=10, marker='+')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.spines[['right', 'top']].set_visible(False)
    return ax

parser = argparse.ArgumentParser(description='''
Spindle - Produces a spindleplot of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

f.setupplot()
spindle(df)
f.savefig(f'{subject}Spindle')


'''
def parse_args(args):
    parser = argparse.ArgumentParser(
       prog='predict.py',
       description='Random Forest Classifier/Regressor with options'
    )
    parser.add_argument('analysis', type=str, help='Regressor or Classifier')
    parser.add_argument('subject', type=str, help='Data name or full filepath')
    parser.add_argument('-n','--n_iter', type=int, help='Number of iterations for bootstrapping', default=10)
    parser.add_argument('--shap_val', action='store_true', help='SHAP interpreted output')
    parser.add_argument('--shap_interact', action='store_true', help='SHAP interaction interpreted output')
    return parser.parse_args(args)

#arguments = ['classifier','speciesCondition.MAM','--shap_val','--shap_interact', '-n=20']
arguments = sys.argv[1:]
args = parse_args(arguments)

# Check if the provided subject is a valid file path
if os.path.isfile(args.subject):
    subject = Path(args.subject).stem
else:
    subject = args.subject

df = f.load(subject)
analysis = args.analysis
shap_val = args.shap_val
shap_interact = args.shap_interact
n_iter = args.n_iter

predict(df, args.analysis, shap_val=args.shap_val, shap_interact=args.shap_interact, n_iter=args.n_iter)
'''
