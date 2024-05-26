#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Volcano - Produces a Volcano plot of a given dataset
''')

parser.add_argument('subject')
parser.add_argument('--change')
parser.add_argument('--sig')
parser.add_argument('--fc')
parser.add_argument('--pval')
parser.add_argument('--annot')

known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

subject = known.get("subject")
change = float(known.get("change")) if known.get("change") else 'Log2FC'
sig = float(known.get("sig")) if known.get("sig") else 'MWW_pval'
fc = float(known.get("fc")) if known.get("fc") else 1
pval = float(known.get("pval")) if known.get("pval") else 0.05
annot = known.get("annot") if known.get("annot") else True

df = f.load(subject)

f.setupplot()
f.volcano(df, change=change, sig=sig, fc=fc, pval=pval, annot=annot)
f.savefig(f'{subject}volcano')
