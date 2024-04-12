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
'''
sys.argv = ['EV.py','geneticsIDRecovery','genusIDRecovery','glitterIDRecovery','kingdomIDRecovery','lipidsIDRecovery','orderIDRecovery','pathwaysallIDRecovery','pathwaystaxoIDRecovery','pathwaysIDRecovery','pciIDRecovery','phylumIDRecovery','potsIDRecovery','psdIDRecovery','PSSIDRecovery','qualityIDRecovery','sleepIDRecovery','speciesIDRecovery','taxoIDRecovery','vepIDRecovery','wolkesIDRecovery']
'''
datasets = {i:pd.read_csv(i, sep='\t', index_col=0) for i in sys.argv[1:]}
print(datasets)

dataset = 'Anthropometrics'
filterdatasets = {}
for dataset in datasets:
    filterdatasets[dataset] = datasets[dataset]
    # If empty or non numeric
    if filterdatasets[dataset].empty or ~(filterdatasets[dataset].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())).any():
        del filterdatasets[dataset]

# Calculate PERMANOVA power
output = pd.Series(index=filterdatasets.keys())
name, ni = 'wolkesIDRecovery', datasets['wolkesIDRecovery']
pval=True
for name, ni in filterdatasets.items():
    tdf = ni.join(meta['Recovery'], how='inner').set_index('Recovery')
    tdf = tdf.loc[tdf.index != 0]
    print(tdf)
    output[name] = f.PERMANOVA(tdf, pval=pval)
power = -output.apply(np.log).sort_values()
f.setupplot(figsize=(3,5))
power.plot.barh()
plt.axvline(x=-np.log(0.05), color="black", linestyle="--")
plt.xlabel('Explained Variance (-log2(pval))')
plt.ylabel(f'{str(time)} dataset')
plt.tight_layout()
f.savefig(f'{str(time)}_EV')
