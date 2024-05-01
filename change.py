#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import os

parser = argparse.ArgumentParser(description='''
Change - Produces a report of the significant feature changes
''')

parser.add_argument('subject', type=str)
parser.add_argument('-a', '--analysis', nargs='+')
parser.add_argument('-c', '--columns', nargs='+')
parser.add_argument('-df2', '--df2')
parser.add_argument('--mult', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

df.index = df.index.astype(str)
output = f.change(df)
print(output)

for table in output:
    f.save(output[table], subject+table)

