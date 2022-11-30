#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''
def setupplot():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["lines.linewidth"] = 0.25
    matplotlib.rcParams["figure.figsize"] = (4, 4)
    matplotlib.rcParams["axes.linewidth"] = 0.25
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['xtick.major.width'] = 0.25
    matplotlib.rcParams['ytick.major.width'] = 0.25
    matplotlib.rcParams['font.family'] = 'Arial'
    

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

def MANNWHIT(df, grouping):
    """
    a, b = MANNWHIT(df.VALUE.to_frame(), df.VISIT == 'BASELINE')
    Has to be between two groups
    """
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection
    #from skbio.stats.composition import multiplicative_replacement as mult
    #df = pd.DataFrame(mult(df), index=df.index, columns=df.columns)
    group1 = df.groupby(grouping).get_group(df[grouping].unique()[0]).drop(grouping, axis=1)
    group2 = df.groupby(grouping).get_group(df[grouping].unique()[1]).drop(grouping, axis=1)
    pvals = []
    for column in df.drop('westernised', axis=1).columns:
        stat , pval = mannwhitneyu(
                group1.loc[:,column],
                group2.loc[:,column],
        )
        pvals.append(pval)
    pvaldf = pd.Series(pvals, index=group1.columns)
    qvaldf = pd.Series(
        fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
        index = pvaldf.index
        )
    return qvaldf

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

def clustermap(df, sig=None, figsize=(4,5)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    #matplotlib.rcParams["font.size"] = 6
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        yticklabels=True,
        xticklabels=True,
    )
    if not sig is None:
        for i, ix in enumerate(g.dendrogram_row.reordered_ind):
            for j, jx in enumerate(g.dendrogram_col.reordered_ind):
                text = g.ax_heatmap.text(
                    j + 0.5,
                    i + 0.5,
                    "*" if sig.iloc[ix, jx] else "",
                    ha="center",
                    va="center",
                    color="black",
                )
                text.set_fontsize(8)
    return g

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
    return df[['PC1', 'PC2']]

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

def PCAplot(df):
    from pca import pca
    model = pca(normalize=True,n_components=0.95)
    model = pca(n_components=0.95)
    results = model.fit_transform(df)
    fig, ax = model.biplot(n_feat=1)

def spindleplot(df, x='PC1', y='PC2', ax=None, palette=None):
    #(df, x, y, palette) = pcoa,'PC1', 'PC2', None
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    if ax is None: fig, ax= plt.subplots()
    centers = df.reset_index().groupby(df.index.names).mean()
    centers.columns=['nPC1','nPC2']
    j = df.join(centers)
    j['colours'] = palette
    i = j.reset_index().index[0]
    for i in j.reset_index().index:
        ax.plot(
                j[['PC1','nPC1']].iloc[i],
                j[['PC2','nPC2']].iloc[i],
                linewidth = 1,
                color = j['colours'].iloc[i],
                zorder=1,
                alpha=0.3
                )
        ax.scatter(j.PC1.iloc[i], j.PC2.iloc[i], color = j['colours'].iloc[i], s=3)
    for i in centers.index:
        ax.text(centers.loc[i,'nPC1']+0.01,centers.loc[i,'nPC2'], s=i, zorder=3)
    ax.scatter(centers.nPC1, centers.nPC2, c='black', zorder=2, s=20, marker='+')
    return ax

def PCspindleplot(df, x='PC1', y='PC2', ax=None):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor, **kwargs
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transf + ax.transData)
        ellipse.set_edgecolor(edgecolor)
        return ax.add_patch(ellipse)
    if not ax: fig, ax= plt.subplots()
    ax = sns.scatterplot(data = df,
        x=x,
        y=y,
        hue=df.index,
        palette=sns.color_palette("husl", df.index.nunique()),
        linewidth=0,
        s=7
    )
    for j, i in enumerate(df.index.unique()):
        confidence_ellipse(
            df.loc[i, x],
            df.loc[i, y],
            ax,
            edgecolor=sns.color_palette("husl", df.index.nunique())[j],
            linestyle='--')
    plt.legend(
        title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small"
    )
    centers = df.groupby(level=0).mean().sort_index()
    centers.columns=['nPC1','nPC2']
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', edgecolor='none', **kwargs):
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor, **kwargs
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transf + ax.transData)
        ellipse.set_edgecolor(edgecolor)
        return ax.add_patch(ellipse)
    if not ax: fig, ax= plt.subplots()
    ax = sns.scatterplot(data = df,
        x=x,
        y=y,
        hue=var,
        palette=sns.color_palette("husl", df[var].nunique()),
        linewidth=0,
        s=7
    )
    for j,i in enumerate(df[var].unique()):
        confidence_ellipse(
            df.loc[df[var] == i][x],
            df[df[var] == i][y],
            ax,
            edgecolor=sns.color_palette("husl", df[var].nunique())[j],
            linestyle='--')
    plt.legend(
        title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small"
    )
    return ax

def rfr(df, var):
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import shap
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1,random_state=1)
    X = df.drop(var, axis=1)
    y = df.xs(var, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_true=y_test, y_pred=y_pred))
    explainer = shap.TreeExplainer(model)
    shaps_values = explainer(X)
    meanabsshap = pd.Series( np.abs(shaps_values.values).mean(axis=0), index=X.columns)
    corrs = [spearmanr(shaps_values.values[:,x], X.iloc[:,x])[0] for x in range(len(X.columns))]
    final = meanabsshap * np.sign(corrs)
    final.fillna(0, inplace=True)
    return final

def rfc(df, var):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import shap
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=1)
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

def circos(nodes, edges):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    ro.globalenv["nodes"] = nodes
    ro.globalenv["edges"] = edges
    ro.r(
        '''
        require(RColorBrewer)
        require(circlize)
        require(ggplot2)
        elements = (unique(c(edges$from, edges$to)))
        set.seed(1)
        gridcol = colorRamp2(c(-1, 0, 1), c("blue", "white", "red"))
        grid.col = gridcol(nodes$X2)
        names(grid.col) = rownames(nodes)
        col_fun = colorRamp2(c(-0.5, 0, 0.5), c("blue", "white", "red"))
        col = col_fun(edges$value)
        names(col) = rownames(edges)
        #svg('../results/fstmp.svg',width=12,height=12)
        circos.par(circle.margin=c(1,1,1,1))
        chordDiagram(edges,
            col = col,
            grid.col=grid.col,
            annotationTrack = "grid",
            preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(edges))))))
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
        '''
    )
    #print(robjects.globalenv["dataR"])

def enterotypes(m, t):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    #utils = importr('utils')
    #package_names = ('DirichletMultinomial', 'dplyr')
    #utils.chooseCRANmirror(ind=70)
    #utils.install_packages(StrVector(package_names))
    mT = m.T.join(t['genus']).groupby('genus').sum()
    unclass = mT[mT.index.str.contains("unclassified")].sum()
    mT = mT[~mT.index.str.contains("unclassified")].T
    mT[-1] = unclass
    df = mT.div(mT.sum(axis=1), axis=0)
    pandas2ri.activate()
    ro.globalenv["df"] = df
    ro.r(
    '''
        if (!require("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        BiocManager::install(version = "3.15")
        BiocManager::install("DirichletMultinomial")
        require(DirichletMultinomial)
        require(dplyr)
        library(parallel)
        genusMat = df
        genusMatT = round(genusMat*10^9)
        genusMatT <- genusMatT[,0:(ncol(genusMatT)-1)]
        set.seed(1)
        #genusFit <- mclapply(1:3, dmn, count=as.matrix(genusMatT), verbose=TRUE)
        genusFit <- sapply(as.matrix(genusMatT), DirichletMultinomial::dmn)
        lplc <- sapply(genusFit, laplace)
        plot(lplc, type="b", xlab="Number of Dirichlet Components",ylab="Model Fit")
        best <- genusFit[[which.min(lplc)]]
        p0 <- fitted(genusFit[[1]], scale=TRUE)
        p3 <- fitted(best, scale=TRUE)
        colnames(p3) <- paste("m", 1:3, sep="")
        (meandiff <- colSums(abs(p3 - as.vector(p0))))
        diff <- rowSums(abs(p3 - as.vector(p0)))
        o <- order(diff, decreasing=TRUE)
        cdiff <- cumsum(diff[o]) / sum(diff)
        df <- head(cbind(Mean=p0[o], p3[o,], diff=diff[o], cdiff), 10)
        heatmapdmn(genusMatT, genusFit[[1]], best, 10, lblwidth = 4000)
        clusterAssigned = apply(genusFit[[3]]@group, 1, function(x) which.max(x))
        clusterAssignedList = split(names(clusterAssigned), clusterAssigned)
        names(clusterAssignedList) = c("ET-Firmicutes","ET-Bacteroides","ET-Prevotella")
        write.csv(stack(clusterAssignedList), 'enterotypes.csv', quote=F)
    '''
    )
    #return ro.globalenv["clusterAssignedList"]
    print(ro.globalenv["clusterAssignedList"])
    print(ro.globalenv["genusMatT"])

def entero(df):
    import pandas as pd
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    df = df.div(df.sum(axis=1), axis=0)
    pandas2ri.activate()
    ro.globalenv["df"] = df
    ro.r(
    '''
        if (!require("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        BiocManager::install(version = "3.15")
        BiocManager::install("DirichletMultinomial")
        require(DirichletMultinomial)
        require(dplyr)
        library(parallel)
        genusMat = df
        genusMatT = round(genusMat*10^9)
        genusMatT <- genusMatT[,0:(ncol(genusMatT)-1)]
        set.seed(1)
        #genusFit <- mclapply(1:3, dmn, count=as.matrix(genusMatT), verbose=TRUE)
        genusFit <- sapply(as.matrix(genusMatT), DirichletMultinomial::dmn)
        lplc <- sapply(genusFit, laplace)
        plot(lplc, type="b", xlab="Number of Dirichlet Components",ylab="Model Fit")
        best <- genusFit[[which.min(lplc)]]
        p0 <- fitted(genusFit[[1]], scale=TRUE)
        p3 <- fitted(best, scale=TRUE)
        colnames(p3) <- paste("m", 1:3, sep="")
        (meandiff <- colSums(abs(p3 - as.vector(p0))))
        diff <- rowSums(abs(p3 - as.vector(p0)))
        o <- order(diff, decreasing=TRUE)
        cdiff <- cumsum(diff[o]) / sum(diff)
        df <- head(cbind(Mean=p0[o], p3[o,], diff=diff[o], cdiff), 10)
        heatmapdmn(genusMatT, genusFit[[1]], best, 10, lblwidth = 4000)
        clusterAssigned = apply(genusFit[[3]]@group, 1, function(x) which.max(x))
        clusterAssignedList = split(names(clusterAssigned), clusterAssigned)
        names(clusterAssignedList) = c("ET-Firmicutes","ET-Bacteroides","ET-Prevotella")
        write.csv(stack(clusterAssignedList), 'enterotypes.csv', quote=F)
    '''
    )
    print(ro.globalenv["clusterAssignedList"])
    print(ro.globalenv["genusMatT"])

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
    plt.rcParams["figure.figsize"] = (2.6,4.3)
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
    return ax

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
    return ax

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

def volcano(lfc, pval, fcthresh=1, pvalthresh=0.05, annot=False, ax=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    if not ax: fig, ax= plt.subplots()
    lpval = pval.apply(np.log10).mul(-1)
    ax.scatter(lfc, lpval, c='black', s=2)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(-1 * np.log10(pvalthresh), color='red', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(fcthresh, color='red', linestyle='-')
    ax.axvline(-fcthresh, color='red', linestyle='-')
    ax.set_ylabel('-log10 p-value')
    ax.set_xlabel('log2 fold change')
    ax.set_ylim(ymin=-0.1)
    x_max = np.abs(ax.get_xlim()).max()
    ax.set_xlim(xmin=-x_max, xmax=x_max)
    sigspecies = lfc.abs().gt(fcthresh) & lpval.abs().gt(-1 * np.log10(pvalthresh))
    sig = pd.concat([lfc.loc[sigspecies], lpval.loc[sigspecies]], axis=1) 
    sig.columns=['x','y']
    if annot: [ax.text(sig.loc[i,'x'], sig.loc[i,'y'], s=i) for i in sig.index]
    return ax

def bar(df):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    #unclass = df[df.index.str.contains("unclassified")].sum()
    #df = df[~df.index.str.contains("unclassified")]
    #df.loc['unclassified'] = unclass
    if df.index.unique().shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values().iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T.sort_index(axis=1)
    #df = df.apply(np.log1p)
    df = df.melt()
    ax = sns.boxplot(data=df, x=df.columns[0], y='value', showfliers=False, boxprops=dict(alpha=.25))
    sns.stripplot(data=df, x=df.columns[0], y='value', size=2, color=".3", ax=ax)
    plt.xlabel(df.columns[0])
    plt.ylabel('Log(Relative abundance)')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()

def box(**kwargs):
    '''
    Should test this!
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statannot
    try: ax = kwargs['ax']
    except: fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(showfliers=False, **kwargs)
    del kwargs['palette']
    sns.stripplot(size=2, color=".3", **kwargs)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    try:
        statannot.add_stat_annotation(
            ax,
            data=data,
            x=x,
            y=y,
            box_pairs=[stats.columns],
            perform_stat_test=False,
            pvalues=kwargs['stats'],
            test_short_name='M.W.W',
            text_format='full',
            )
    except:
        pass
    return ax

def fancybox(df,x,y,p, ax=None):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
    ax = sns.boxplot(data=df, x=x,y=y,  showfliers=False)
    sns.stripplot(data=df, x=x,y=y, size=2, c=".3", jitter=False, ax=ax)
    sns.pointplot(data=df, x=x, y=y, hue=p, c='.3', ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    return ax

def to_edges(df, thresh=0.1):
    import pandas as pd
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source')

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

def joseplot2(t1, s1, t2, s2):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    df = t1.join(s1).join(t2).join(s2)
    df.columns=['t1','s1','t2','s2']
    df['s'] = df['s1'] + df['s2']
    sns.scatterplot(data=df, x='t1', y='t2', c='s', size=1)
    #fig = px.scatter(data, x= 't1', y='t2', hover_name=data.index)
    return fig

def scatterplot(df1, df2, cor):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import numpy as np
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

def newcurve(df, mapping):
    from scipy import stats
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df = df.apply(np.log1p).T
    df.loc['other'] = df[~df.index.isin(mapping.index)].sum()
    df = df.drop(df[~df.index.isin(mapping.index)].index).T
    grid = np.linspace(df.index.min(), df.index.max(), num=500)
    df1poly = df.apply(lambda x: np.polyfit(x.index, x.values, 3), axis=0).T
    df1polyfit = df1poly.apply(lambda x: np.poly1d(x)(grid), axis=1)
    griddf = pd.DataFrame( np.row_stack(df1polyfit), index=df1polyfit.index, columns=grid)
    griddf.clip(lower=0, inplace=True)
    if griddf.shape[0] > 20:
        griddf.loc["other"] = griddf.loc[
            griddf.T.sum().sort_values(ascending=False).iloc[21:].index
        ].sum()
    griddf = griddf.loc[griddf.T.sum().sort_values().tail(20).index].T
    griddf.sort_index(axis=1).plot.area(stacked=True, color=mapping.to_dict(), figsize=(9, 2), ax=ax)
    plt.xlim(0, 35)
    plt.legend(
        title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small"
    )
    plt.ylabel("Log(Relative abundance)")
    plt.tight_layout()
    plotdf = df.cumsum(axis=1).stack().reset_index()
    sns.scatterplot(data=plotdf.sort_values(0), x='Days after birth', y=0, s=10, linewidth=0, ax=ax)

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

def density(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    df.T.plot.density()
    plt.ylim(bottom = 0)

def crossval(model, X, y, fold=10):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut 
    from sklearn.model_selection import cross_validate
    import matplotlib.pyplot as plt
    import numpy as np
    kf = KFold(fold)
    tprs = []
    aucs = []
    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')
    for train, test in kf.split(X,y):
        results = model.fit(X.iloc[train], y.iloc[train])
        y_score = model.predict_proba(X.iloc[test])
        fpr, tpr, _ = roc_curve(y.iloc[test], y_score[:, 1])
        aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    mean_auc = auc(base_fpr, mean_tprs)
    std_auc = np.std(aucs)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.plot(base_fpr, mean_tprs, 'b', label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label=r"$\pm$ 1 std. dev.",)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")

def aucroc(model, X, y, ax=None, colour=None):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
    y_score = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_score)
    AUC = auc(fpr, tpr)
    if colour is None:
        ax.plot(fpr, tpr, label=r"AUCROC = %0.2f" % AUC )
    else:
        ax.plot(fpr, tpr, color=colour, label=r"AUCROC = %0.2f" % AUC )
    ax.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    return ax

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

def clusterplot(G):
    import networkx as nx
    nx.draw(
            G,
            with_labels=True,
            node_color="b",
            node_size=50
            )
    
def annotateplot(G, group):
    import matplotlib.pyplot as plt
    from itertools import count
    import networkx as nx
    nx.set_node_attributes(G, group, "group")
    #nx.set_node_attributes(G, pd.Series(labeler.labels_, index=G.nodes), "group")
    groups = set(nx.get_node_attributes(G, 'group').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[n]['group']] for n in nodes]
    pos = nx.spring_layout(G)
    #pos= nx.spring_layout(G, with_labels=True, node_size=50)
    ax = nx.draw_networkx_edges(G, pos, alpha=0.2)
    #nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
    ax = nx.draw_networkx_nodes(G, pos, node_color=group, node_size=20, cmap=plt.cm.jet)
    #nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    plt.colorbar(ax)
    return ax
    
def clusterdendrogram(G):
    import networkx as nx
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    d           = { 0: [1, 'd'], 1: ['a', 'b', 'c'], 'a': [], 'b': [], 'c': [], 'd': []}
    G           = nx.DiGraph(d)
    nodes       = G.nodes()
    leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
    inner_nodes = [ n for n in nodes if G.out_degree(n) > 0 ]
    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(d[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += d[v]
        subtree[u] = sorted(children & leaves)
    inner_nodes.sort(key=lambda n: len(subtree[n]))
    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = d[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(subtree[x] + subtree[y])
            i, j = index[tuple(subtree[x])], index[tuple(subtree[y])]
            Z.append([i, j, float(len(subtree[n])), len(z)]) # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1
    dendrogram(Z, labels=leaves)

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

def mww(df, mult=False):
    ''' index needs to be grouping element '''
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection 
    import itertools
    import pandas as pd
    combs = list(itertools.combinations(df.index.unique(), 2))
    outdf = pd.DataFrame(
        [mannwhitneyu(df.loc[i[0]], df.loc[i[1]])[1] for i in combs],
        columns = df.columns,
        index = combs
        )
    if mult:
        outdf = pd.DataFrame(
            fdrcorrection(outdf.values.flatten())[1].reshape(outdf.shape),
            columns = outdf.columns,
            index = outdf.index
            )
    return outdf

def logfc(df):
    ''' index needs to be grouping element, df needs to be zero free '''
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import fdrcorrection 
    from itertools import combinations as c
    import pandas as pd
    import numpy as np
    combs = list(c(df.index.unique(), 2))
    outdf = pd.DataFrame(np.array(
        [df.loc[i[0]].mean().div(df.loc[i[1]].mean()) for i in combs]),
        columns = df.columns,
        index = combs
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
