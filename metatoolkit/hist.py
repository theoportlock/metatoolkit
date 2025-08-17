#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Hist - Produces a Histplot of a given dataset
''')

def hist(df, **kwargs):
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    kwargs['column'] = kwargs.get('column', 'sig')

    sns.histplot(data=df, x=kwargs['column'], ax=kwargs['ax'])
    ax = kwargs['ax']
    if ax is not None:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax

parser.add_argument('subject')
parser.add_argument('-c', '--column')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.pop("subject")
if os.path.isfile(subject):
    subject = Path(subject).stem

df = pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

plt.figure()
hist(df, **known)
plt.savefig(f'results/{subject}hist.svg')

