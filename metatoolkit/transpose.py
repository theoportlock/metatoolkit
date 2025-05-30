#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='''
Transpose - Produces a transposition of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
if not os.path.isfile(subject):
    subject = f'results/{subject}.tsv'
df = pd.read_csv(subject, sep='\t', index_col=0)

outdf = df.T

subject = Path(subject).stem
outdf.to_csv(f'results/{subject}_T.tsv', sep='\t')

