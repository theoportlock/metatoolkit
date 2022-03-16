#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Plotting an MSP PCoA 
'''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skbio
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

msp_taxonomy = pd.read_csv("taxo.csv", index_col=0)
msp_samples = pd.read_csv("msp.csv", index_col=0)
samples_metadata = pd.read_csv('metadata.csv', index_col=0)

# Join taxonomic information
taxaType = "species"
samples_taxonomy = msp_samples.join(msp_taxonomy[taxaType], how='inner').groupby(taxaType).sum().T

# Compute PCoA
Ar_dist = distance.squareform(distance.pdist(samples_taxonomy, metric="braycurtis"))
DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
results = PCoA.samples.copy()
samples_taxonomy['PC1'], samples_taxonomy['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values

# Plot metadata label
samples_taxonomyMetadata = samples_taxonomy.join(samples_metadata, how='inner')
sns.scatterplot(data = samples_taxonomyMetadata, x='PC1', y='PC2', hue='Gender', palette='colorblind')
plt.show()

'''
# Compute best cluster
n_clusters = range(2,20)
models = [KMeans(n_clusters=i).fit(samples_taxonomy[['PC1', 'PC2']]) for i in n_clusters]
sscores = pd.Series([silhouette_score(samples_taxonomy[['PC1', 'PC2']], i.labels_) for i in models], index=n_clusters)
print(sscores)
samples_taxonomy['Cluster'] = models[sscores.reset_index(drop=True).idxmax()].labels_

# Plot best cluster
sns.scatterplot(data = samples_taxonomy, x='PC1', y='PC2', hue='Cluster', palette='colorblind')
plt.show()
'''

#plt.tight_layout()
#plt.savefig("results/PCoA.pdf")
plt.show()
