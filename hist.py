#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='''
Hist - Produces a Histplot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('-c', '--column')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = f.load(subject)

f.setupplot()
f.hist(df, **known)
f.savefig(f'{subject}hist')
