#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics plotting
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
    
def heatmap(df, sig=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option("use_inf_as_na", True)
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
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values(ascending=False).iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index]
    norm = df.T.div(df.sum(axis=0), axis=0)
    ax = norm.plot(kind='bar',stacked=True, width=0.9, cmap='tab20', ylim=(0,1))
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    return ax

def abund(df):
    import matplotlib.pyplot as plt
    df = df.T
    if df.shape[0] > 20:
        df.loc['other'] = df.loc[df.T.sum().sort_values(ascending=False).iloc[21:].index].sum()
    df = df.loc[df.T.sum().sort_values().tail(20).index].T
    ax = df.plot(kind='bar',stacked=True, width=0.9, cmap='tab20')
    plt.legend(title='Gene', bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    return ax

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
    df = df.apply(np.log1p).T
    df.loc['other'] = df[~df.index.isin(mapping.index)].sum()
    df = df.drop(df[~df.index.isin(mapping.index)].index)
    df.T.plot.area(stacked=True, color=mapping, figsize=(9, 2))
    plt.legend(loc='center left', bbox_to_anchor=(0, -0.1))
    plt.xlim(0, 35)
    plt.ylabel("Log(Relative abundance)")

def aucroc(model, X, y, ax=None, colour=None):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
    y_score = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
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
