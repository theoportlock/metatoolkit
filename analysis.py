#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''

def ANCOM(df, perm=False):
    """
    sig = ancom(df)
    """
    import pandas as pd
    from skbio.stats.composition import ancom
    from scipy.stats import mannwhitneyu
    from itertools import combinations
    from itertools import permutations
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
            [ancom(pd.concat([df.loc[i[0]], df.loc[i[1]]]), pd.concat([df.loc[i[0]], df.loc[i[1]]]).index.to_series())[0]['Reject null hypothesis'] for i in combs],
            columns = df.columns,
            index = combs,
            )
    return outdf

def PERMANOVA(df, meta):
    import pandas as pd
    from skbio.stats.distance import permanova
    from skbio import DistanceMatrix
    from scipy.spatial import distance
    beta = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    pvals = permanova(DistanceMatrix(beta, beta.index), meta)
    return pvals

def corr(df1, df2):
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import fdrcorrection
    #df1 = df1.loc[:,df1.nunique() > 5]
    #df2 = df2.loc[:,df2.nunique() > 5]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df.values)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    pvaldf.fillna(1, inplace=True)
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return cordf, qvaldf

def PCOA(df):
    import pandas as pd
    import numpy as np
    import skbio
    from scipy.spatial import distance
    df = df.loc[:, df.sum() != 0]
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    results = PCoA.samples.copy()
    df['PC1'], df['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return df[['PC1', 'PC2']]

def PCA(df):
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df[['PC1', 'PC2']]

def StandardScale(df):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def TSNE(df):
    import numpy as np
    from sklearn.manifold import TSNE
    results = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def UMAP(df):
    import umap
    from sklearn.preprocessing import StandardScaler
    scaledDf = StandardScaler().fit_transform(df)
    reducer = umap.UMAP()
    results = reducer.fit_transform(scaledDf)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def SOM(df):
    from sklearn_som.som import SOM
    som = SOM(m=3, n=1, dim=2)
    som.fit(df)
    return som

def NMDS(df):
    ''' Needs to be a beta diveristy or correlation matrix (square) '''
    import pandas as pd
    from sklearn.manifold import MDS
    from scipy.spatial import distance
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    mds = MDS(n_components = 2, metric = False, max_iter = 500, eps = 1e-12, dissimilarity = 'precomputed')
    results = mds.fit_transform(BC_dist)
    df['PC1'], df['PC2'] = results[:,0], results[:,1]
    return df[['PC1', 'PC2']]

def RFC(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify=y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    print('AUCROC =', roc_auc_score(y, model.predict_proba(X)[:, 1]))
    print(confusion_matrix(y_test, y_pred))
    return model

def SHAP_bin(X,model):
    import numpy as np
    from scipy.stats import spearmanr
    import pandas as pd
    import shap
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series(
            np.abs(shaps_values.values[:,:,0]).mean(axis=0),
            index=X.columns
            )
    corrs = [spearmanr(shaps_values.values[:,x,1], X.iloc[:,x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return final

def SHAP_interact(X,model):
    import numpy as np
    from scipy.stats import spearmanr
    import pandas as pd
    import shap
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]):
        vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    return final

def shannon(df):
    import pandas as pd
    from skbio.diversity.alpha import shannon
    df = pd.DataFrame(df.agg(shannon, axis=1), columns=['Shannon Diversity'])
    return df

def evenness(df):
    import pandas as pd
    from skbio.diversity.alpha import pielou_e
    df = pd.DataFrame(df.agg(pielou_e, axis=1), columns=['Pielou Evenness'])
    return df

def richness(df):
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(df.agg(np.count_nonzero, axis=1), columns=['Richness'])
    return df

def beta(df):
    import pandas as pd
    from scipy.spatial import distance
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    return BC_dist

def to_edges(df, thresh=0.1):
    import pandas as pd
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source')

def mult(df):
    import pandas as pd
    from skbio.stats.composition import multiplicative_replacement as mul
    return pd.DataFrame(mul(df), index=df.index, columns=df.columns)

def taxofunc(msp, taxo, short=False, mult=False):
    import pandas as pd
    from skbio.stats.composition import multiplicative_replacement as mul
    m, t = msp.copy(), taxo.copy()
    m = m.loc[m.sum(axis=1) != 0, m.sum(axis=0) != 0]
    if mult:
        m = pd.DataFrame(mul(m), index=m.index, columns=m.columns)
    taxolevels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    t['superkingdom'] = 'k_' + t['superkingdom']
    t['phylum'] = t[['superkingdom', 'phylum']].apply(lambda x: '|p_'.join(x.dropna().astype(str).values), axis=1)
    t['class'] = t[['phylum', 'class']].apply(lambda x: '|c_'.join(x.dropna().astype(str).values), axis=1)
    t['order'] = t[['class', 'order']].apply(lambda x: '|o_'.join(x.dropna().astype(str).values), axis=1)
    t['family'] = t[['order', 'family']].apply(lambda x: '|f_'.join(x.dropna().astype(str).values), axis=1)
    t['genus'] = t[['family', 'genus']].apply(lambda x: '|g_'.join(x.dropna().astype(str).values), axis=1)
    t['species'] = t[['genus', 'species']].apply(lambda x: '|s_'.join(x.dropna().astype(str).values), axis=1)
    t["species"] = t["species"].str.replace(" ", "_").str.replace('/.*','')
    mt = m.join(t[taxolevels])
    df = pd.concat([mt.groupby(taxolevels[i]).sum() for i in range(len(taxolevels))])
    #df = df.loc[~df.index.str.contains('unclassified')]
    if short:
        df.index = df.T.add_prefix("|").T.index.str.extract(".*\|([a-z]_.*$)", expand=True)[0]
    #df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    return df

def to_network(edges):
    import networkx as nx
    G = nx.from_pandas_edgelist(edges)
    return G

def cluster(G):
    from networkx.algorithms.community.centrality import girvan_newman as cluster
    import pandas as pd
    #clust = nx.clustering(G)
    communities = cluster(G)
    node_groups = []
    for com in next(communities):
        node_groups.append(list(com))
    df = pd.DataFrame(index=G.nodes, columns=['Group'])
    for i in range(pd.DataFrame(node_groups).shape[0]):
        tmp = pd.DataFrame(node_groups).T[i].to_frame().dropna()
        df.loc[tmp[i], 'Group'] = i
    return df.Group

def sig(df, mult=False, perm=False):
    ''' index needs to be grouping element '''
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection 
    from itertools import combinations
    from itertools import permutations
    import numpy as np
    import pandas as pd
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(
        [mannwhitneyu(df.loc[i[0]], df.loc[i[1]])[1] for i in combs],
        columns = df.columns,
        index = combs
        ).T
    if mult:
        outdf = pd.DataFrame(
            fdrcorrection(outdf.values.flatten())[1].reshape(outdf.shape),
            columns = outdf.columns,
            index = outdf.index
            )
    return outdf

def lfc(df, mult=False, perm=False):
    ''' index needs to be grouping element, df needs to be zero free '''
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu
    from itertools import combinations
    from itertools import permutations
    from statsmodels.stats.multitest import fdrcorrection 
    from skbio.stats.composition import multiplicative_replacement as m
    if mult: df = pd.DataFrame(m(df), index=df.index, columns=df.columns)
    combs = list(combinations(df.index.unique(), 2))
    if perm: combs = list(permutations(df.index.unique(), 2))
    outdf = pd.DataFrame(np.array(
        [df.loc[i[0]].mean().div(df.loc[i[1]].mean()) for i in combs]),
        columns = df.columns,
        index = combs
        ).T.apply(np.log2)
    return outdf

def clr(df, axis=0):
    '''
    Centered Log Ratio
    Aitchison, j. (1982)
    '''
    from numpy import log1p
    tmp = log1p(df)
    f = lambda x: x - x.mean()
    z = tmp.apply(f, axis=1-axis)
    return z

def CLR_normalize(pd_dataframe):
    """
    Centered Log Ratio
    Aitchison, J. (1982). 
    The statistical analysis of compositional data. 
    Journal of the Royal Statistical Society: 
    Series B (Methodological), 44(2), 139-160.
    """
    d = pd_dataframe
    d = d+1
    step1_1 = d.apply(np.log, 0)
    step1_2 = step1_1.apply(np.average, 0)
    step1_3 = step1_2.apply(np.exp)
    step2 = d.divide(step1_3, 1)
    step3 = step2.apply(np.log, 0)
    return(step3)

def cm(X, y, labels=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    import numpy as np
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    tprs, aucs = [],[]
    base_fpr = np.linspace(0, 1, 101)
    #from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    #cm = confusion_matrix(y_true, y_pred)
    if not labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    return cm

if __name__ == '__main__':
    import doctest
    doctest.testmod()
