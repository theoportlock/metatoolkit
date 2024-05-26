#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
'''
import functions as f
import sys

# load data
file = sys.argv[1]
cats = f.load(file)

# Remove cases where there are only one label or 5 categories
cats = cats.loc[:, cats.sum() != cats.shape[0]]
cats = cats.loc[:, cats.sum() >= 5]

# Remove Nos
cats = cats.loc[:, ~cats.columns.str.contains('No')]

# calculate fisher exact
out = f.fisher(cats)
out.index.set_names(['source', 'target'], inplace=True)

# remove duplicates
out.drop_duplicates(inplace=True)

# save fisher results
f.save(out, f'{file}fisher')
