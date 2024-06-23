#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f
import pandas as pd

parser = argparse.ArgumentParser(description='''
shap - compute a SHAP value for each sample based on features in AI model''')

parser.add_argument('analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

subject = known.get("subject")

df = f.load(subject)
with open(f'../results/{known.get("subject")}predict.pkl', 'rb') as file:
    model = pickle.load(file)
output = f.explain(df, model, **known)
print(output.to_string())

f.save(output, f'{subject}Shap')
