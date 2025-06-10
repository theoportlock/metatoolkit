#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def load(subject):
    if os.path.isfile(subject):
        return pd.read_csv(subject, sep='\t', index_col=0)
    return pd.read_csv(f'results/{subject}.tsv', sep='\t', index_col=0)

parser = argparse.ArgumentParser(description='''
Upset - Produces an upset plot of indecies of multiple datasets
''')
parser.add_argument('datasets', nargs='+')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

dfs = known.get("datasets")
alldfs = {df:load(df).index for df in dfs}

f.upset(alldfs)
plt.savefig(f'results/upset.svg')

