#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='Group - Groups a dataset')
parser.add_argument('subject')
parser.add_argument('-t', '--type')
parser.add_argument('-o', '--output')

known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")
output = known.get("output")
type = known.get("type")

known.pop("subject")
df = f.load(subject)
out = f.group(df, **known)
print(out)

if known.get("output"):
    f.save(out, output)
else:
    f.save(out, subject+type)
