#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


parser = argparse.ArgumentParser(description='''
Line - Produces a Lineplot of a given dataset
''')
parser.add_argument('subject')
parser.add_argument('-df2', '--df2', type=str, default='meta', help='categorical data to label lineplot with')
parser.add_argument('-x')
parser.add_argument('-y')
parser.add_argument('--hue')
parser.add_argument('--units')
parser.add_argument('--logy', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

# Load data
subject = known.get('subject')
if subject is not None and os.path.isfile(subject): subject = Path(subject).stem
if subject is None:
    raise ValueError("The 'subject' argument must be provided and not None.")
df = pd.read_csv(subject, sep='\t', index_col=0)

df2_path = known.get('df2')
if df2_path is None:
    raise ValueError("The 'df2' argument must be provided and not None.")
df2 = pd.read_csv(df2_path, index_col=0, sep='\t')

# Merge metadata
plotdf = df.join(df2)
print(plotdf)

# Load variables
units = known.get('units')
x = known.get("x")
y = known.get("y")
hue = known.get("hue")
logy = known.get("logy")

# Sort
plotdf = plotdf.sort_values(x)

# Plot and save
sns.lineplot(data=plotdf,
             x=x,
             y=y,
             units=units,
             hue=hue,
             estimator=None)
if logy: plt.yscale('log')
plt.savefig(f'results/{subject}_line.svg')

