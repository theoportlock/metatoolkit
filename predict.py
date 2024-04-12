#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functions as f

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

df = f.load(subject)
output = f.predict(df, **known)
print(output)
with open(f'../results/{known.get("subject")}predict.pkl', 'wb') as file: pickle.dump(output[0], file)  
with open(f'../results/{known.get("subject")}performance.txt', 'w') as of: of.write(output[1])
