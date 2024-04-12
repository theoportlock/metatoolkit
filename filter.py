#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Filter')
parser.add_argument('subject')
parser.add_argument('-lt', type=float, help='threshold value upper bound for filtering')
parser.add_argument('-gt', type=float, help='threshold value lower bound for filtering')
parser.add_argument('-c', '--column', type=str, help='column for thesholding')
parser.add_argument('-n', '--name', type=str, help='name for filtering')
parser.add_argument('-rf', '--rowfilt', type=str, help='regex for index filtering')
parser.add_argument('-cf', '--colfilt', type=str, help='regex for column filtering')
parser.add_argument('-q', '--query', type=str, help='Pandas custom query')
parser.add_argument('-m', '--min_unique', type=int, help='Minimum number')
parser.add_argument('-fdf', '--filter_df')
parser.add_argument('-fdfx', '--filter_df_axis', type=int)
parser.add_argument('-absgt', type=float)
parser.add_argument('-p', '--prevail', type=float)
parser.add_argument('-a', '--abund', type=float)
parser.add_argument('--numeric_only', action='store_true')
parser.add_argument('--nonzero', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop('subject')
df = f.load(subject)

if known.get("filter_df"):
    known['filter_df'] = f.load(known.get("filter_df"))

output = f.filter(df, **known)
print(output)
if output is not None:
    f.save(output, subject + 'filter')
