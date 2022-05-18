#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''
#matplotlib.use('Agg')
#sns.set_context("paper", rc={"font.size":24,"axes.titlesize":24,"axes.labelsize":24, "axes.ticklabelssize": 24})
def setupplot():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.size"] = 12
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (4, 4)

def fc(df, basename):
    """
    Calculate the fold change of a dataframe from the baseline combined with statistics

    :param df: DataFrame
    :param b: str
    :return: (df, df)

    >>> fc(2, 3)
    5
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
    #df = df.loc[:, (df.groupby(level=[0, 1]).nunique() > 3).all()]
    logfcdf = df.div(
        df.xs(basename, level=0).droplevel(0),
        level=2
    ).apply(np.log2)
    logfcdf = logfcdf.groupby(level=[0, 1]).mean()
    logfcdf = logfcdf.drop(basename, level=0).T
    def appl(df, baseline):
        result = pd.Series(dtype='float64')
        for i in df.columns:
            result[i] = mannwhitneyu(
            baseline.xs(df.index[0][-2])[i],
            df[i]
            )[1]
        return result
    baseline = df.xs(basename, level=0)
    notbaseline = df.drop(basename, level=0)
    pvaldf = notbaseline.groupby(level=[0, 1]).apply(appl, (baseline)).T
    return logfcdf.T, pvaldf.T

def nfc(df, basename):
    ''' Visit, Arm, Subject'''
    import numpy as np
    from scipy.stats import mannwhitneyu
    df = df.loc[:, (df.groupby(level=[0, 1]).nunique() > 3).all()]
    logfcdf = df.div(
        df.xs(basename, level=0).droplevel(0),
        level=2
    ).apply(np.log2)
    return logfcdf

def wilcoxgroup(df, basename):
    import pandas as pd
    from scipy.stats import mannwhitneyu
    def appl(df, baseline):
        result = pd.Series(dtype='float64')
        baselinevar = df.reset_index()[groupnam].unique()[0]
        for i in df.columns:
            result[i] = mannwhitneyu(baseline.loc[baselinevar,i],df[i])[1]
        return result
    baseline = df.xs(basename, level=0)
    notbaseline = df.drop(basename, level=0)
    pvaldf = notbaseline.groupby(level=[0, 1]).apply(appl, (baseline)).T
    pvaldf.replace([np.inf, -np.inf], np.nan, inplace=True)
    pvaldf.dropna(inplace=True)

def wilcox(df, basename):
    import pandas as pd
    from scipy.stats import mannwhitneyu
    df.sort_index(inplace=True)
    def appl(df, baseline):
        result = pd.Series(dtype='float64')
        for i in df.columns:
            result[i] = mannwhitneyu(baseline[i], df[i])[1]
        return result
    baseline = df.loc[basename]
    notbaseline = df.drop(basename)
    pvaldf = notbaseline.groupby(level=0).apply(appl, (baseline)).T
    pvaldf.replace([np.inf, -np.inf], np.nan, inplace=True)
    pvaldf.dropna(inplace=True)

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
    pvaldf.fillna(1, inplace=True)
    qvaldf = pd.DataFrame(
        fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return cordf, qvaldf

def smallcorr(df):
    import pandas as pd
    from scipy.stats import spearmanr
    rho = df.corr()
    pval = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: "".join(["*" for t in [0.01, 0.05, 0.1] if x <= t]))
    rho = rho.round(2).astype(str) + p
    return rho

def pcorr(df1, df2):
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import fdrcorrection
    df1 = df1.loc[:,df1.nunique() > 5]
    df2 = df2.loc[:,df2.nunique() > 5]
    df = df1.join(df2, how='inner')
    cordf = df.corr()
    cordf = cordf.loc[df1.columns, df2.columns]
    return cordf

def heatmap(df, sig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option("use_inf_as_na", True)
    #plt.rcParams["figure.figsize"] = df.shape
    df = df.T
    if not sig is None:
        sig = sig.T
        df = df[(sig < 0.05).sum(axis=1) > 0]
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
    if not sig is None:
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
                text.set_fontsize(8)
    #g.set_aspect(1)
    plt.setp(g.get_xticklabels(), rotation=40, ha="right")
    #plt.tight_layout()
    return g

def clustermap(df, sig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option("use_inf_as_na", True)
    df.dropna(inplace=True)
    df = df.T
    if not sig is None:
        sig = sig.T
        df = df[(sig < 0.05).sum(axis=1) > 0]
        sig = sig.loc[df.index]
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        #linewidths=0.1,
        #linecolor='gray'
    )
    if not sig is None:
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
                text.set_fontsize(8)

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
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
    return df

def TSNE(df):
    import numpy as np
    from sklearn.manifold import TSNE
    ndf = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(df)
    return df

def UMAP(df):
    import umap
    scaledDf = StandardScaler().fit_transform(df.T)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaledDf)
    return df

def SOM(df):
    from sklearn_som.som import SOM
    som = SOM(m=3, n=1, dim=2)
    som.fit(df)
    return df

def PCAplot(df):
    from pca import pca
    model = pca(normalize=True,n_components=0.95)
    model = pca(n_components=0.95)
    results = model.fit_transform(df)
    fig, ax = model.biplot(n_feat=1)

def PCplot(df, var):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    plt.rcParams["figure.figsize"] = (8,8)
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
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
        ellipse.set_edgecolor(edgecolor)
        return ax.add_patch(ellipse)
    ax = sns.scatterplot(data = df,
        x='PC1',
        y='PC2',
        hue=var,
        palette='colorblind'
    )
    for j,i in enumerate(df[var].unique()):
        confidence_ellipse(
            df.loc[df[var] == i]['PC1'],
            df[df[var] == i]['PC2'],
            ax,
            edgecolor=sns.color_palette()[j],
            linestyle='--')
    return ax

def rfr(df, var):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from scipy.stats import spearmanr
    import shap
    import numpy as np
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=1)
    X = df.drop(var, axis=1)
    y = df.xs(var, axis=1)
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
    return final

def rfc(df, var):
    from sklearn import metrics
    from sklearn.decomposition import PCA
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import pandas as pd
    model = RandomForestClassifier(n_estimators=500, random_state=1)
    X = df.drop(var, axis=1)
    y = df.xs(var, axis=1)
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
    return final

def circos(fc, pval):
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
    df = df.T
    plt.rcParams["figure.figsize"] = (14,4)
    #df.sort_index(axis=1, inplace=True)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    df.loc['unclassified'] = unclass
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values(ascending=False).iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index]
    norm = df.T.div(df.sum(axis=0), axis=0)
    ax = norm.plot(kind='bar',stacked=True, width=0.9, cmap='tab20', ylim=(0,1))
    plt.legend(title='Taxonomy', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

def abund(df):
    import matplotlib.pyplot as plt
    df = df.T
    plt.rcParams["figure.figsize"] = (14,4)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    #df.sort_index(axis=1, inplace=True)
    df.loc['unclassified'] = unclass
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values(ascending=False).iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T
    ax = df.plot(kind='bar',stacked=True, width=0.9, cmap='tab20')
    plt.legend(title='Gene', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

def shannon(df):
    import pandas as pd
    from skbio.diversity.alpha import shannon
    df = pd.DataFrame(df.agg(shannon, axis=1), columns=['Shannon Diversity Index'])
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

def volcano(fc, pval):
    import pandas as pd
    from bioinfokit import analys, visuz
    df = pd.merge(fc.to_frame(), pval.to_frame(), right_index=True, left_index=True)
    df.columns = ['logFC', 'P-value']
    visuz.GeneExpression.volcano(
        df=df.dropna().reset_index(),
        lfc="logFC",
        pv="P-value",
        pv_thr=(0.5, 0.5),
        lfc_thr=(0.1, 0.1),
        geneid="index",
        sign_line=True,
        show=True,
        plotlegend=True,
        legendpos="upper right",
        genenames="deg",
    )

def bar(df):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    plt.rcParams["figure.figsize"] = (5,5)
    unclass = df[df.index.str.contains("unclassified")].sum()
    df = df[~df.index.str.contains("unclassified")]
    df.loc['unclassified'] = unclass
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values().iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T.sort_index(axis=1)
    df = df.apply(np.log1p).melt()
    ax = sns.boxplot(data=df, x=df.columns[0], y='value', showfliers=False, boxprops=dict(alpha=.25))
    sns.stripplot(data=df, x=df.columns[0], y='value', size=2, color=".3", ax=ax)
    plt.xlabel(df.columns[0])
    plt.ylabel('Log(Relative abundance)')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

def box(df,x,y):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["font.size"] = 14
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (5,5)
    ax = sns.boxplot(data=df, x=x,y=y,  showfliers=False)
    sns.stripplot(data=df, x=x,y=y, size=2, color=".3", ax=ax)
    plt.xlabel(x)
    plt.ylabel(y)
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

def joseplot(fc, t1, t2, pval=None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    t1 = fc.xs(t1)
    t2 = fc.xs(t2)
    t1 = t1.reindex_like(t2)
    t2 = t2.reindex_like(t1)
    data = t1.mean().to_frame().join(t2.mean().to_frame().add_suffix('t2'))
    data.columns=['t1', 't2']
    #sns.scatterplot(data=data, x='t1', y='t2', size=1)
    fig = px.scatter(data, x= 't1', y='t2', hover_name=data.index)
    return fig

def scatterplot(df1, df2, cor):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    df = df1.join(df2, how='inner')
    for i in cor.index:
        for j in cor.columns:
            sns.lmplot(data=np.log1p(metaData.join(taxoMspMeta)), x=i, y=j)
            plt.tight_layout()
            plt.show()
    #fig = px.scatter(data, x= 't1', y='t2', hover_name=data.index)
    #return fig

def pairgrid(df):
    g = sns.PairGrid(df)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.lmplot)

def curve(df):
    from scipy import stats
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    grid = np.linspace(df.index.min(), df.index.max(), num=500)
    df1poly = df.apply(lambda x: np.polyfit(x.index, x.values, 3), axis=0).T
    df1polyfit = df1poly.apply(lambda x: np.poly1d(x)(grid), axis=1)
    griddf = pd.DataFrame(
        np.row_stack(df1polyfit), index=df1polyfit.index, columns=grid
    )
    griddf.clip(lower=0, inplace=True)
    griddf = griddf.apply(np.log1p)
    if griddf.shape[0] > 20:
        griddf.loc["other"] = griddf.loc[
            griddf.T.sum().sort_values(ascending=False).iloc[21:].index
        ].sum()
    griddf = griddf.loc[griddf.T.sum().sort_values().tail(20).index].T
    griddf.plot.area(stacked=True, cmap="tab20")
    plt.legend(
        title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small"
    )
    plt.ylabel("Log(Relative abundance)")
    plt.tight_layout()

def nocurve(df, mapping):
    from scipy import stats
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    mapping = mapping.sort_index()
    #fig, ax = plt.subplots(figsize=(5, 7))
    df = df.apply(np.log1p).T
    #if df.shape[0] > 20:
    #    df.loc["other"] = df.loc[
    #        df.T.sum().sort_values(ascending=False).iloc[21:].index
    #    ].sum()
    #df = df.loc[df.T.sum().sort_values().tail(20).index]
    df.loc['other'] = df[~df.index.isin(mapping.index)].sum()
    df = df.drop(df[~df.index.isin(mapping.index)].index)
    #df = df.loc[df.sum(axis=1).sort_values().index]
    df.T.plot.area(stacked=True, color=mapping, figsize=(9, 2))
    #plt.legend(
        #title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small"
    #)
    plt.legend(loc='center left', bbox_to_anchor=(0, -0.1))
    #plt.ylim(0, 0.0008)
    plt.xlim(0, 35)
    #plt.legend('',frameon=False)
    plt.ylabel("Log(Relative abundance)")
    #plt.tight_layout()

def taxofunc(msp, taxo):
    import pandas as pd
    taxolevels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    taxo['superkingdom'] = 'k_' + taxo['superkingdom']
    taxo['phylum'] = taxo[['superkingdom', 'phylum']].apply(lambda x: '|p_'.join(x.dropna().astype(str).values), axis=1)
    taxo['class'] = taxo[['phylum', 'class']].apply(lambda x: '|c_'.join(x.dropna().astype(str).values), axis=1)
    taxo['order'] = taxo[['class', 'order']].apply(lambda x: '|o_'.join(x.dropna().astype(str).values), axis=1)
    taxo['family'] = taxo[['order', 'family']].apply(lambda x: '|f_'.join(x.dropna().astype(str).values), axis=1)
    taxo['genus'] = taxo[['family', 'genus']].apply(lambda x: '|g_'.join(x.dropna().astype(str).values), axis=1)
    taxo['species'] = taxo[['genus', 'species']].apply(lambda x: '|s_'.join(x.dropna().astype(str).values), axis=1)
    taxo["species"] = taxo["species"].str.replace(" ", "_")
    msptaxo = msp.join(taxo[taxolevels])
    df = pd.concat([msptaxo.groupby(taxolevels[i]).sum() for i in range(len(taxolevels))])
    df = df.loc[~df.index.str.contains('unclassified')]
    return df

def density(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    df.T.plot.density()
    plt.ylim(bottom = 0)

def plotextremes(df):
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    df[zscore(df.abs()) > 1].sort_values().plot.bar()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
