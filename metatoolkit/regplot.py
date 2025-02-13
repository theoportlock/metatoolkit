#!/usr/bin/env python
# -*- coding: utf-8 -*-

import seaborn as sns
import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Regplot - Produces a Regplot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('--hue')
parser.add_argument('--logy', action='store_true')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
df = f.load(subject)
logy = known.get("logy"); known.pop("logy")
hue = known.get('hue')

f.setupplot()
if hue:
    x = known.get('x')
    y = known.get('y')
    sns.regplot(data = df, x=x, y=y, color='red', scatter=False)
    sns.scatterplot(data = df, x=x, y=y, s=2, hue=hue, legend=False)
else:
    f.regplot(df, **known)

if logy: plt.yscale('log')
plt.tight_layout()
f.savefig(f'{subject}regplot')

