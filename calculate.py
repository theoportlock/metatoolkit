#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='''
Calculate - compute a value for each sample based on features
''')

parser.add_argument('analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject"); known.pop("subject")
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)

analysis = known.get('analysis')
output = f.calculate(analysis, df)
print(output)
f.save(output, subject + known.get("analysis"))
