#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='''
Stratify - Stratifies a dataframe according to the column of another dataframe (commonly metadata)
''')

parser.add_argument('subject')
parser.add_argument('level')
parser.add_argument('--df2')
known, unknown = parser.parse_known_args()
known = {k: v for k, v in vars(known).items() if v is not None}
unknown = eval(unknown[0]) if unknown != [] else {}

# IF IS FILE THEN LOAD BUT NOT THEN DO THE OTHER LOAD
df = f.load(known.get("subject"))
meta = f.load(known.get("df2")) if known.get("df2") else f.load('meta')
level = known.get('level')

output = f.stratify(df, meta, level)
print(output)
f.save(output, f'{known.get("subject")}{level}')
