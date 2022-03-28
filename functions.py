#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''
import pandas as pd
def fc(df, basename):
    import numpy as np
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection
    df = df.loc[:, (df.groupby(level=[0, 1]).nunique() > 3).all()]
    logfcdf = df.div(
        df.xs(basename, level=0).droplevel(0),
        level=2
    ).apply(np.log2)
    logfcdf = logfcdf.groupby(level=[0, 1]).median().T
    logfcdf = logfcdf.T.drop(basename, level=0).T
    def appl(df, baseline):
        result = pd.Series(dtype='float64')
        baselinevar = df.reset_index()['Group'].unique()[0]
        for i in df.columns:
            result[i] = mannwhitneyu(baseline.loc[baselinevar,i],df[i])[1]
        return result
    baseline = df.xs(basename, level=0)
    notbaseline = df.drop(basename, level=0)
    pvaldf = notbaseline.groupby(level=[0, 1]).apply(appl, (baseline)).T
    pvaldf.replace([np.inf, -np.inf], np.nan, inplace=True)
    pvaldf.dropna(inplace=True)
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[0].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return logfcdf, pvaldf

def corr(df1, df2):
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import fdrcorrection
    df1 = df1.loc[:,df1.nunique() > 5]
    df2 = df2.loc[:,df2.nunique() > 5]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df.values)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return cordf, qvaldf

def heatmap(df, sig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option("use_inf_as_na", True)
    plt.rcParams["figure.figsize"] = (4,4)
    if not sig is None: df = df[(sig < 0.05).sum(axis=1) > 0]
    sig = sig.loc[df.index]
    g = sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )
    for tick in g.get_yticklabels(): tick.set_rotation(0)
    annot=pd.DataFrame(index=sig.index, columns=sig.columns)
    annot[(sig < 0.0005) & (df > 0)] = '+++'
    annot[(sig < 0.005) & (df > 0)] = '++'
    annot[(sig < 0.05) & (df > 0)] = '+'
    annot[(sig < 0.0005) & (df < 0)] = '---'
    annot[(sig < 0.005) & (df < 0)] = '--'
    annot[(sig < 0.05) & (df < 0)] = '-'
    annot[sig >= 0.05] = ''
    for i, ix in enumerate(df.index):
        for j, jx in enumerate(df.columns):
            text = g.text(
                j + 0.5,
                i + 0.5,
                annot.values[i,j],
                ha="center",
                va="center",
                color="black",
            )
            #text.set_fontsize(8)
    plt.tight_layout()

def clustermap(df, sig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option("use_inf_as_na", True)
    df.dropna(inplace=True)
    if not sig is None: df = df[(sig < 0.05).sum(axis=1) > 0]
    sig = sig.loc[df.index]
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )
    for tick in g.ax_heatmap.get_yticklabels(): tick.set_rotation(0)
    annot=pd.DataFrame(index=sig.index, columns=sig.columns)
    annot[(sig < 0.0005) & (df > 0)] = '+++'
    annot[(sig < 0.005) & (df > 0)] = '++'
    annot[(sig < 0.05) & (df > 0)] = '+'
    annot[(sig < 0.0005) & (df < 0)] = '---'
    annot[(sig < 0.005) & (df < 0)] = '--'
    annot[(sig < 0.05) & (df < 0)] = '-'
    annot[sig >= 0.05] = ''
    for i, ix in enumerate(df.index):
        for j, jx in enumerate(df.columns):
            text = g.ax_heatmap.text(
                j + 0.5,
                i + 0.5,
                annot.values[i, j],
                ha="center",
                va="center",
                color="black",
            )
            #text.set_fontsize(8)

def PCOA(df):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import skbio
    from scipy.spatial import distance
    df = df.loc[:, df.sum() != 0]
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    results = PCoA.samples.copy()
    df['PC1'], df['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return df

def PCA(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import plotly.express as px
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    scaledDf = StandardScaler().fit_transform(df.T)
    pca = PCA()
    results = pca.fit_transform(scaledDf)
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df

def PCplot(df, var):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    ax = sns.scatterplot(data = df,
        x='PC1',
        y='PC2',
        hue=var,
        #palette='colorblind'
    )
    for j,i in enumerate(df[var].unique()):
        confidence_ellipse(
            df.loc[df[var] == i]['PC1'],
            df[df[var] == i]['PC2'],
            ax,
            #edgecolor=sns.color_palette()[j],
            #edgecolor='auto',
            linestyle='--')
    return ax

def rfr(df, var):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    model = ExtraTreesRegressor(n_estimators=500, n_jobs=-1,random_state=1)
    X = testdf.drop(var, axis=1)
    y = testdf.xs(var, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(r2_score(y_true=y_test, y_pred=y_pred))
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series( np.abs(shaps_values.values).mean(axis=0), index=X.columns)
    corrs = [spearmanr(shaps_values.values[:,x], X.iloc[:,x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return model, final

def rfc(df, var):
    from sklearn import metrics
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import pandas as pd
    model = RandomForestClassifier(n_estimators=500, random_state=1)
    X = testdf.drop(var, axis=1)
    y = testdf.xs(var, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=y_pred))
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series( np.abs(shaps_values.values).mean(axis=0), index=X.columns)
    corrs = [spearmanr(shaps_values.values[:,x], X.iloc[:,x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return model, final

def circos(df):
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    robjects.r('''
    require(RColorBrewer)
    require(circlize)
    require(ggplot2)
    circosInput <- read.csv('fsnewedges.csv')
    changes <- read.csv('nnnnodes.csv', row.names=1)
    elements = (unique(c(circosInput$from, circosInput$to)))
    set.seed(1)
    gridcol = colorRamp2(c(-1, 0, 1), c("blue", "white", "red"))
    grid.col = gridcol(changes$X2)
    names(grid.col) = rownames(changes)
    col_fun = colorRamp2(c(-0.5, 0, 0.5), c("blue", "white", "red"))
    col = col_fun(circosInput$value)
    names(col) = rownames(circosInput)
    svg('../results/fstmp.svg',width=12,height=12)
    circos.par(circle.margin=c(1,1,1,1))
    chordDiagram(circosInput,
        col = col,
        grid.col=grid.col,
        annotationTrack = "grid",
        preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(circosInput))))))
    circos.track(track.index = 1, panel.fun = function(x, y) {
        circos.text(CELL_META$xcenter,
            CELL_META$ylim[1],
            CELL_META$sector.index,
            facing = "clockwise",
            niceFacing = TRUE,
            adj = c(0, 0.5))
            },
        bg.border = NA)
    circos.clear() 
    dev.off()
    ''')

def venn(df1, df2, df3):
    from matplotlib_venn import venn3
    from itertools import combinations
    def combs(x): return [c for i in range(1, len(x)+1) for c in combinations(x,i)]
    comb = combs([df1, df2, df3])
    result = []
    for i in comb:
        if len(i) > 1:
            result.append(len(set.intersection(*(set(j.columns) for j in i))))
        else:
            result.append(len(i[0].columns))
    venn3(subsets = result)

def relabund(df):
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,4)
    #df.sort_index(axis=1, inplace=True)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    df.loc['unclassified'] = unclass
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values().iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index]
    norm = df.T.div(df.sum(axis=0), axis=0)
    norm.plot(kind='bar',stacked=True, width=0.9, cmap='tab20', ylim=(0,1))
    plt.legend(title='Taxonomy', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.tight_layout()

def abund(df):
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12,4)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    #df.sort_index(axis=1, inplace=True)
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values().iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T
    df.plot(kind='bar',stacked=True, width=0.9, cmap='tab20')
    plt.legend(title='Gene', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.tight_layout()

def shannon(df):
    from skbio.diversity.alpha import shannon
    df = pd.DataFrame(df.agg(shannon, axis=1), columns=['Shannon Diversity Index'])
    return df

def richness(df):
    import numpy as np
    df = pd.DataFrame(df.agg(np.count_nonzero, axis=1), columns=['Species Richness'])
    return df

def beta(df):
    return None

def volcano(df):
    from bioinfokit import analys, visuz
    visuz.gene_exp.volcano(df=df.reset_index(), lfc='logFC', pv='P-value', geneid='index', sign_line=True, show=True, plotlegend=True, legendpos='upper right', genenames='deg')

def comparebar(df):
    return None

def bar(df):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (5,5)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    df.loc['unclassified'] = unclass
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values().iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T.sort_index(axis=1)
    df = df.apply(np.log1p).melt()
    ax = sns.boxplot(data=df, x=df.columns[0], y='value', showfliers=False)
    sns.stripplot(data=df, x=df.columns[0], y='value', size=2, color=".3", ax=ax)
    plt.xlabel(df.columns[0])
    plt.ylabel('Log(Relative abundance)')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

def to_network(df):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    edges = taxoMspMetaFc.stack().stack().to_frame()
    sigedges = taxoMspMetaPval.stack().to_frame()
    fin = edges[sigedges < 0.2].dropna()
    #fin = edges.loc[np.abs(cor.stack()) > 0.4].dropna()
