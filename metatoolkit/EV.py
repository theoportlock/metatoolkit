#!/usr/bin/env python
import numpy as np
import sys
import functions as f
import matplotlib.pyplot as plt
import pandas as pd

# load and format data
meta = f.load('meta')

dataset_names = sys.argv
if len(sys.argv)<2:
    raise SystemExit('Please add dataset(s) as arguments to this command')
filterdatasets = {i:pd.read_csv(i, sep='\t', index_col=0) for i in sys.argv[1:]}

# Calculate PERMANOVA power
output = pd.Series(index=filterdatasets.keys())
pval=True
for name, tdf in filterdatasets.items():
    #tdf = tdf.loc[tdf.index != 0]
    print(tdf)
    output[name] = f.PERMANOVA(tdf, pval=pval)
power = -output.apply(np.log).sort_values()
f.setupplot(figsize=(3,5))
power.plot.barh()
plt.axvline(x=-np.log(0.05), color="black", linestyle="--")
plt.xlabel('Explained Variance (-log2(pval))')
#plt.ylabel(f'{str(time)} dataset')
plt.tight_layout()
f.savefig(f'EV')
