#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import functions as f
import os

parser = argparse.ArgumentParser(description='Calculate - compute a value for each sample based on features')
parser.add_argument('analysis')
parser.add_argument('subject')
known = parser.parse_args()
known = {k: v for k, v in vars(known).items() if v is not None}

analysis = known.get("analysis"); known.pop('analysis')
subject = known.get("subject"); known.pop('subject')
if os.path.isfile(subject): subject = Path(subject).stem
df = f.load(subject)
print(df)

output = f.predict(analysis, df, **known)
print(output)
with open(f'../results/{known.get("subject")}predict.pkl', 'wb') as file: pickle.dump(output[0], file)  
with open(f'../results/{known.get("subject")}performance.txt', 'w') as of: of.write(output[1])

'''
# Save from multiple objects
with open("models.pckl", "wb") as f:
    for model in models:
         pickle.dump(model, f)

# And load
models = []
with open("models.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
        except EOFError:
            break
'''
