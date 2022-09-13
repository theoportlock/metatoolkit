#!/usr/bin/env python
import gc
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

# set up cluster and workers
client = Client(n_workers=4, threads_per_worker=1, memory_limit='64GB') 
client

# Load data
gene_info = dd.read_csv('../data/oralGene.tsv', sep='\t')
gct_data = dd.read_csv("../data/oralgct.tsv", sep='\t', assume_missing=True)
#gut_msp_data['#gene_id'] = gut_msp_data['#gene_id'].astype(np.int64) 

# Map to catalogues
## KEGG
kegg = dd.read_csv('../data/oralKegg.csv')
kegg.columns = ['gene_id','gene_name','ko', 'bitscore']
replacedbykegg = gct_data.merge(kegg, how='inner', left_on='#gene_id', right_on='gene_id')
fmtcols = replacedbykegg.columns[replacedbykegg.columns.isin(gct_data.columns)].to_list()
fmtcols.append('ko')
sumofsm = replacedbykegg[fmtcols].groupby('ko').sum()
del replacedbykegg
del kegg
gc.collect()
sumofsm.to_csv("oralkegg.csv", single_file=True)
del sumofsm

pfam = dd.read_csv('../data/oralPfamAllTab.csv', assume_missing=True)
replacedbygenename = gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbypfam = replacedbygenename.merge(pfam, how='inner', left_on='gene_name', right_on='gene_name')
fmtcols = replacedbypfam.columns[replacedbypfam.columns.isin(gct_data.columns)].to_list()
fmtcols.append('pfam_name')
sumofsm = replacedbypfam[fmtcols].groupby('pfam_name').sum()
del replacedbypfam, pfam
gc.collect()
sumofsm.to_csv("oralpfam.csv", single_file=True)
del sumofsm

## CARD
card = pd.read_csv('../data/oralCard.tsv', sep='\t')
card["gene_name"] = card.ORF_ID.str.split(expand=True)[0]
card["gene_name"] = card["gene_name"].str[0:-2]
replacedbygenename = gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbycard = replacedbygenename.merge(card, how='inner', left_on='gene_name', right_on='gene_name')
del replacedbygenename
fmtcols = replacedbycard.columns[replacedbycard.columns.isin(gct_data.columns)].to_list()
fmtcols.append('Best_Hit_ARO')
sumofsm = replacedbycard[fmtcols].groupby('Best_Hit_ARO').sum()
del replacedbycard, card
gc.collect()
sumofsm.to_csv("oralcard.csv", single_file=True)
del sumofsm

## CAzyme
cazyme = pd.read_csv('../data/oralCazy.csv', sep='\t')
replacedbycazyme = gct_data.merge(cazyme, how='inner', left_on='#gene_id', right_on='gene_id')
fmtcols = replacedbykegg.columns[replacedbykegg.columns.isin(gct_data.columns)].to_list()
fmtcols.append('CAZyme')
sumofsm = replacedbycazyme[fmtcols].groupby('CAZyme').sum()
del replacedbycazyme, cazyme
gc.collect()
sumofsm.to_csv("oralcazyme.csv", single_file=True)
del sumofsm

client.close()
