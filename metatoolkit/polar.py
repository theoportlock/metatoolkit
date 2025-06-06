#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='''
Polar - Produces a Polarplot of a given dataset
''')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
df = f.load(subject)

f.setupplot()
output = f.polar(df)
f.savefig(f'{subject}polar')
