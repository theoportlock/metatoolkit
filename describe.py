#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
parser.add_argument('subject')
parser.add_argument('-p', '--pval', type=float)
parser.add_argument('-c', '--change')
parser.add_argument('-s', '--sig')
parser.add_argument('-r', '--corr', action='store_true')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

df = f.load(subject)
output = f.describe(df, **known)
print(output.to_string())
f.save(output, f'{subject}describe')
