#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Finds species and data changes and correlates
'''
from skbio.stats.composition import multiplicative_replacement
import functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load data ---------------------------------
meta = pd.read_csv("../data/meta.csv", index_col=0)
msp = pd.read_csv("../data/msp.csv", index_col=0)
gene = pd.read_csv("../data/gct.csv", index_col=0)
taxo = pd.read_csv("../data/taxo.csv", index_col=0)
data = pd.read_csv('../data/data.csv', index_col=0)

variables = ['VISIT', 'ARM', 'SUBJID']

taxoMsp = functions.taxofunc(msp.T, taxo, short=True).T
taxoMsp = pd.DataFrame(
    multiplicative_replacement(taxoMsp), index=msp.index, columns=msp.columns
)

metaTaxoMsp = taxoMsp.join(meta[variables], how='inner').set_index(variables)
metaData = data.join(meta[variables]).dropna().set_index(variables)

# Diversity ----------------------------
taxoMsp = msp.copy()
taxoMsp = taxoMsp[taxoMsp.columns[~taxoMsp.columns.str.contains('unclassified')]]
variables = ['VISIT', 'ARM', 'SUBJID']
metaTaxoMsp = taxoMsp.join(meta[variables], how='inner').set_index(variables)

sns.catplot(
    data=functions.shannon(msp).reset_index(),
    x="VISIT",
    y="Shannon Diversity Index",
    col="ARM",
    hue="SUBJID",
    kind="point",
)
plt.savefig('../results/shannon.svg')
plt.show()

sns.catplot(
    data=functions.richness(msp).reset_index(),
    x="VISIT",
    y="Richness",
    hue="ARM",
    col="Site",
    kind="point",
)
plt.savefig('../results/richness.svg')
plt.show()

# Correlation ---------------------------------
cor = functions.pcorr(metaData, metaTaxoMsp)
sigcor = cor.loc[:, (cor.abs() > 0.4).any()]
functions.heatmap(sigcor)
edges = cor.stack().to_frame()
edges.reset_index(inplace=True) 
edges.columns = ['from','to','value']
sigedges = edges[edges.abs() > 0.5].unstack(level=0).T.droplevel(0).T.dropna(thresh=1)
edges.set_index('from', inplace=True)
edges.to_csv('../results/edges.csv')

# Fold Change ---------------------------------
fcMetaTaxoMsp, pvalMetaTaxoMsp = functions.fc(metaTaxoMsp[cor.columns], 'BASELINE')
fcMetaData, pvalMetaData = functions.fc(metaData[cor.index], 'BASELINE')
'''
fcMetaTaxoMsp, pvalMetaTaxoMsp = functions.fc(metaTaxoMsp, 'BASELINE')
fcMetaData, pvalMetaData = functions.fc(metaData, 'BASELINE')
'''
functions.heatmap(taxoMspMetaFc); plt.show()
functions.heatmap(metaDataFc); plt.show()

nodes = pd.concat([metaDataFc.T["WEEK04", "ACTIVE"], taxoMspMetaFc.T["WEEK04", "ACTIVE"].fillna(0)])
nodes.name = 'value'
nodes.to_csv('../results/nodes.csv')

# Circos ------------------------------
function.circos(edges, nodes)

# BIOMARKER ----------------------------
biomarkers = [
'Veillonella parvula',
'Veillonella dispar',
'Veillonella atypica',
'Streptococcus parasanguinis',
'Streptococcus salivarius',
'Haemophilus parainfluenzae']

# species specific -------------------
var='Strep'
sliced = taxoMspMeta.loc[:, taxoMspMeta.columns.str.contains(var)].sort_index(level=0)
sliced = taxoMspMeta.loc[:, biomarkers].sort_index(level=0)
ax = sns.pointplot(
    data=sliced.stack().to_frame().reset_index(),
    x="VISIT",
    y=0,
    hue="level_3",
)
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
plt.legend(title='Taxonomy', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
plt.ylabel('Relative abundance')
plt.tight_layout()
plt.savefig(f'../results/{var}.svg')
plt.show()

ax = sns.clustermap(
    sliced.groupby(level=[0, 1]).mean().sort_index(level=1).T,
    cmap="vlag",
    center=0,
    col_cluster=False,
)
plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=40, ha="right")
plt.show()

plt.rcParams["figure.figsize"] = (5,5)
df = df.apply(np.log1p).melt()
ax = sns.boxplot(data=df, x=df.columns[0], y='value', showfliers=False)
sns.stripplot(data=df, x=df.columns[0], y='value', size=2, color=".3", ax=ax)
plt.xlabel(df.columns[0])
plt.ylabel('Log(Relative abundance)')
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

# Random Forest------------------------------
sdf = functions.rfr(taxoMspMeta.join(metaData['D-Lactate']).dropna(), "D-Lactate")
functions.plotextremes(sdf)
