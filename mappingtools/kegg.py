#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Theo Portlock
Creates a pseudo GCT from MSP table
'''
import numpy as np
import pandas as pd
import dask.dataframe as dd

msp = dd.read_csv("msp.csv")
gene = pd.read_csv("gutGene.tsv", sep='\t',index_col=0)
kegg = pd.read_csv('gutKegg.csv',index_col=0)

pgct = gene.join(msp)
pgct.to_csv("../results/pgct.csv",  single_file=True)

pgct = dd.read_csv('../results/pgct.csv').set_index('gene_name')
pgct.join(kegg, how='inner').groupby('ko').sum().to_csv('../results/pkegg.csv')
