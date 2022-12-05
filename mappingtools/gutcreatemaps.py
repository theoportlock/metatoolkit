#!/usr/bin/env python
import gc
import numpy as np
import itertools
import seaborn as sns
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client

# set up cluster and workers
client = Client(n_workers=8, threads_per_worker=1, memory_limit='1.5GB') 
client = Client() 
client

# Data location
file_gut_gct_data="../data/gct.tsv"
file_gene_info='../../data/gutGene.tsv'

file_kegg='../../data/gutKegg.csv'
file_patric='../../data/gutPatric.csv'
file_antismash='../../data/gutAntismash.tsv'
file_pfam='../../data/gutPfamDesc.csv'
file_patric='../../data/gutPatric.csv'
file_card='../../data/gutCard.tsv'
file_cazyme='../../data/gutCard.tsv'

# Load data
gene_info = dd.read_csv(file_gene_info, sep='\t')
gut_gct_data = dd.read_csv(file_gut_gct_data, sep='\t', assume_missing=True)
gut_gct_data['#gene_id'] = gut_gct_data['#gene_id'].astype(np.int64) 

# Map to catalogues
## KEGG
kegg = dd.read_csv(file_kegg, dtype={'Unnamed: 0':int,'gene_name':str, 'ko':str, 'bitscore': str})
#kegg.columns = ['gene_id','gene_name','ko', 'bitscore']
keggMapping = gut_gct_data.merge(kegg[['gene_id','ko']], how='inner', left_on='#gene_id', right_on='gene_id')
#fmtcols = replacedbykegg.columns[replacedbykegg.columns.isin(gut_gct_data.columns)].to_list()
#fmtcols.append('ko')
#sumofsm = replacedbykegg[fmtcols].groupby('ko').sum()
#del keggMapping
#del kegg
gc.collect()
#sumofsm.to_csv("kegg.csv",  single_file=True)
keggMapping = keggMapping.drop(['#gene_id', 'gene_id'], axis=1)
final = keggMapping.groupby('ko').sum()
# still need to update this with the dask scheduler on 127.0.0.1:8787

final.to_csv("../data/gutKeggMapping.tsv", sep='\t', single_file=True)
del sumofsm

## antismash
antismash = pd.read_table(file_antismash)
antismash.columns = ['gene_name','sm','id']
replacedbygenename = gut_gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbyantismash = replacedbygenename.merge(antismash, how='inner', left_on='gene_name', right_on='gene_name')
fmtcols = replacedbykegg.columns[replacedbykegg.columns.isin(gut_gct_data.columns)].to_list()
fmtcols.append('sm')
sumofsm = replacedbyantismash[fmtcols].groupby('sm').sum()
del replacedbyantismash, antismash
gc.collect()
sumofsm.to_csv("antismash.csv",  single_file=True)
del sumofsm

## pfam 
pfam = dd.read_csv(file_pfam, assume_missing=True)
replacedbygenename = gut_gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbypfam = replacedbygenename.merge(pfam, how='inner', left_on='gene_name', right_on='gene_name')
fmtcols = replacedbykegg.columns[replacedbykegg.columns.isin(gut_gct_data.columns)].to_list()
fmtcols.append('pfam_name')
sumofsm = replacedbypfam[fmtcols].groupby('pfam_name').sum()
del replacedbypfam, pfam
gc.collect()
sumofsm.to_csv("pfam.csv",  single_file=True)
del sumofsm

## patric 
patric = dd.read_csv(file_patric)
replacedbygenename = gut_gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbypatric = replacedbygenename.merge(patric, how='inner', left_on='gene_name', right_on='igc2_id')
del replacedbygenename, patric
gc.collect()
replacedbypatric.to_csv("../data/gutPatricMapping.csv", single_file=True)
del replacedbypatric

## CARD
card = pd.read_csv(file_card, sep='\t')
card["gene_name"] = card.ORF_ID.str.split(expand=True)[0]
card["gene_name"] = card["gene_name"].str[0:-2]
replacedbygenename = gut_gct_data.merge(gene_info, how='inner', left_on='#gene_id', right_on='gene_id')
replacedbycard = replacedbygenename.merge(card, how='inner', left_on='gene_name', right_on='gene_name')
del replacedbygenename
fmtcols = replacedbycard.columns[replacedbycard.columns.isin(gut_gct_data.columns)].to_list()
fmtcols.append('Best_Hit_ARO')
sumofsm = replacedbycard[fmtcols].groupby('Best_Hit_ARO').sum()
del replacedbycard, card
gc.collect()
sumofsm.to_csv("../data/gutCardMapping.csv",  single_file=True)
del sumofsm

## CAzyme
cazyme = pd.read_csv(file_cazyme, sep='\t')
replacedbycazyme = gut_gct_data.merge(cazyme, how='inner', left_on='#gene_id', right_on='gene_id')
fmtcols = [col for col in replacedbycazyme.columns.values if 'FMT' in col]
fmtcols.append('CAZyme')
sumofsm = replacedbycazyme[fmtcols].groupby('CAZyme').sum()
del replacedbycazyme, cazyme
gc.collect()
sumofsm.to_csv("cazyme.csv",  single_file=True)
del sumofsm

client.close()
