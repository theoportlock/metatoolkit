#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics analysis
'''
def fc(df, basename):
    import pandas as pd
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
        baselinevar = df.reset_index()['ARM'].unique()[0]
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
        fdrcorrection(pvaldf.values.flatten())[0].reshape(pvaldf.shape),
        index = pvaldf.index,
        columns = pvaldf.columns)
    return cordf, qvaldf

def heatmap(df, sig=pd.Series(dtype='float64'):
    import seaborn as sns
    sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )

def clustermap(df, sig=pd.Series(dtype='float64')):
    import seaborn as sns
    sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.1,
        linecolor='gray'
    )

def PCOA(df):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import skbio
    from scipy.spatial import distance
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)
    results = PCoA.samples.copy()
    df['PC1'], df['PC2'] = results.iloc[:,0].values, results.iloc[:,1].values
    return df[['PC1','PC2']]

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
    return df[['PC1','PC2']]

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
        palette='colorblind'
    )
    for i in df.var.unique():
        confidence_ellipse(df.loc['PC1', var], df['PC2', var], ax)

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
