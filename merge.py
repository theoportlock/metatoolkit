#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Merge - Combines datasets')
parser.add_argument('datasets', nargs='+')
parser.add_argument('-j', '--join')
parser.add_argument('-a', '--append', action='store_true')
parser.add_argument('-o', '--output')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

dfs = known.get("datasets")
alldfs = [f.load(df) for df in dfs]
join=known.get("join") if known.get("join") else 'inner'
output = known.get("filename")

result = f.merge(datasets=alldfs, join=join)
print(result)

if output:
    f.save(result, output)
else:
    f.save(result, "".join(dfs))
