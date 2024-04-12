#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='''
Calculate - compute a value for each sample based on features
''')

parser.add_argument('analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

df = f.load(known.get('subject'))
analysis = known.get('analysis')
output = f.calculate(analysis, df)
print(output)
f.save(output, known.get("subject") + known.get("analysis"))
