#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='''
Heatmap - Produces a heatmap of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

f.setupplot()
subject = known.get("subject")
df = f.load(subject)
f.spindle(df)
f.savefig(f'{subject}spindle')
