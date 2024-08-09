#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

def taxo_summary(df):
    count = df.columns.str.replace(".*\|", "", regex=True).str[0].value_counts()
    return count

subject = known.get("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

output = taxo_summary(df)
print(output.to_string())
f.save(output, f'{subject}Taxocount')
