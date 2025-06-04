#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def splitter(df, df2, column):
    output = {}
    if df2 is None:
        df2 = df.copy()
    for level in df2[column].unique():
        merge = df.loc[:, df.columns[df.columns != column]].join(df2.loc[df2[column] == level, column], how='inner').drop(column, axis=1)
        output[level] = merge
    return output

parser = argparse.ArgumentParser(description='''
Splitter - splits dataframes according to the values in a defined column''')
parser.add_argument('subject')
parser.add_argument('column')
parser.add_argument('--df2', required=False)

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

df = load(known.get('subject'))
df2 = load(known.get('df2')) if known.get("df2") else None
col = known.get('column')
subject = known.get('subject')

output = splitter(df, df2, col)
print(output)

for level in output:
    save(output[level], f'{subject}{col}{level}')
