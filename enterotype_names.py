import functions as f
import pandas as pd
import numpy as np

meta = f.load('meta')
meta.index = meta.index.astype(str)

# enterotype description
df = f.load('enterotypes')
taxo = f.load('taxo')
strat = f.stratify(taxo, df, 'ind')
strat.index = strat.index.astype(str)

# for naming enterotypes
ch = f.change(strat)
spec = pd.DataFrame(columns=strat.columns)
for c in ch:
    spec.loc[c] = ch[c]['MWW_qval'].apply(np.log).mul(-1) * np.sign(ch[c]['Log2FC'])
spec.index = spec.index.str.split('vs', expand=True)
fspec = spec.groupby(level=0).mean()
maxvals = fspec.idxmax(axis=1).values
print(maxvals)
names = fspec.idxmax(axis=1).str.replace('.*\|','',regex=True).to_dict()
strat.index = strat.index.map(names)
df.ind = df.ind.astype(str).map(names)
f.save(df, 'namedenterotypes')

# relabund
f.setupplot()
plotdf = strat.loc[:, maxvals].apply(np.log1p)
f.abund(plotdf.groupby(level=0).mean())
f.savefig('enterorelabund')

# spindle
f.setupplot()
pcoa = f.calculate('pcoa', strat)
f.spindle(pcoa)
f.savefig('enterospindle')
