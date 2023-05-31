#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Scripts for metagenomics plotting
'''
def setupplot():
    import matplotlib
    #matplotlib.use('Agg')
    linewidth = 0.25
    matplotlib.rcParams['grid.color'] = 'lightgray'
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["lines.linewidth"] = linewidth
    matplotlib.rcParams["figure.figsize"] = (4, 4)
    matplotlib.rcParams["axes.linewidth"] = linewidth
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['xtick.major.width'] = linewidth
    matplotlib.rcParams['ytick.major.width'] = linewidth
    matplotlib.rcParams['font.family'] = 'Arial'
    matplotlib.rcParams['axes.axisbelow'] = True

def pointheatmap(df, ax=None, size_scale=300):
    import matplotlib.pyplot as plt
    df.columns.name='x'
    df.index.name='y'
    vals = df.unstack()
    vals.name='size'
    fvals = vals.to_frame().reset_index()
    x, y, size= fvals.x, fvals.y, fvals['size']
    if ax is None: fig, ax= plt.subplots()
    x_labels = x.unique()
    y_labels = y.unique()
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
    )
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    plt.grid()
    return ax
    
def heatmap(df, sig=None, ax=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    if ax is None: fig, ax= plt.subplots()
    pd.set_option("use_inf_as_na", True)
    if not sig is None:
        df = df[(sig < 0.05).sum(axis=1) > 0]
        sig = sig.loc[df.index]
    g = sns.heatmap(
        data=df,
        square=True,
        cmap="vlag",
        center=0,
        yticklabels=True,
        xticklabels=True,
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
    plt.setp(g.get_xticklabels(), rotation=40, ha="right")
    return g

def clustermap(df, sig=None, figsize=(4,5), **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    g = sns.clustermap(
        data=df,
        cmap="vlag",
        center=0,
        figsize=figsize,
        dendrogram_ratio=(0.25, 0.25),
        yticklabels=True,
        xticklabels=True,
        **kwargs,
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
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right")
    return g

def extremes(df, n=50):
    import pandas as pd
    return pd.concat([df.sort_values().head(n),
                      df.sort_values().tail(n)])

def levene(df):
    import pandas as pd
    from scipy.stats import levene
    output = pd.Series()
    for col in df.columns: 
        output[col] = levene(*[df.loc[cat,col] for cat in df.index.unique()])[1]
    return output

def shapiro(df):
    import pandas as pd
    from scipy.stats import shapiro
    output = pd.DataFrame()
    for col in df.columns: 
        for cat in df.index.unique():
            output.loc[col,cat] = shapiro(df.loc[cat,col])[1]
    return output

def spindleplot(df, x='PC1', y='PC2', ax=None, palette=None):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    if ax is None: fig, ax= plt.subplots()
    centers = df.groupby(df.index).mean()
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
    ro.r('''
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

def upset(dfdict):
    from upsetplot import UpSet, from_contents
    intersections = from_contents(dfdict) 
    upset = UpSet(intersections)
    upset.plot()

def sigCorrPlot(corr, thresh=0.5, **kwargs):
    import pandas as pd
    sigcorrs = pd.concat(
            [cor.gt(thresh).sum(),
             cor.lt(-thresh).sum()],
            axis=1,
            keys=['gt','lt']
            )
    sigcorrs['no_change'] = sigcorrs['gt'].add(sigcorrs['lt']).sub(cor.shape[0]).abs()
    f.abund(sigcorrs)

def vsnnorm(df):
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    # import the 'vsn' package from Bioconductor
    r_data_matrix = numpy2ri.numpy2ri(df)
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('vsn')
    vsn = rpackages.importr('vsn')
    # convert the data matrix to an R matrix object
    r_data_matrix = robjects.r['matrix'](robjects.FloatVector(df.flatten()), nrow=len(data_matrix), ncol=len(data_matrix[0]))
    # perform VSN normalization
    vsn_normalized_matrix = vsn.vsn(r_data_matrix)
    # convert the normalized matrix back to a Python numpy array
    normalized_matrix = np.array(vsn_normalized_matrix)
    return(normalized_matrix)

def relabund(df, **kwargs):
    import matplotlib.pyplot as plt
    try: ax = kwargs['ax']
    except: fig, ax = plt.subplots(figsize=(4, 4))
    if df.shape[1] > 20:
        df['other'] = df[df.sum().sort_values(ascending=False).iloc[19:].index].sum(axis=1)
    df = df[df.sum().sort_values().tail(20).index]
    df = df.T.div(df.sum(axis=1), axis=1).T
    df.plot(kind='bar',stacked=True, width=0.9, cmap='tab20', ylim=(0,1), ax=ax)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax

def abund(df, **kwargs):
    # Merge with relabund
    import matplotlib.pyplot as plt
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    if df.shape[1] > 20:
        df['other'] = df[df.sum().sort_values(ascending=False).iloc[19:].index].sum(axis=1)
    df = df[df.sum().sort_values().tail(20).index]
    df.plot(kind='bar', stacked=True, width=0.9, cmap='tab20', **kwargs)
    plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left', fontsize='small')
    plt.ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def volcano(lfc, pval, fcthresh=1, pvalthresh=0.05, annot=False, ax=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    if not ax: fig, ax= plt.subplots()
    lpval = pval.apply(np.log10).mul(-1)
    ax.scatter(lfc, lpval, c='black', s=0.5)
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

def bar(*args, **kwargs):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    kwargs['ax'] = plt.subplots()[1] if not kwargs.get('ax') else kwargs.get('ax')
    df = args[0].copy()
    if df.columns.shape[0] > 20:
        df['other'] = df[df.mean().sort_values().iloc[21:].index].sum(axis=1)
    df = df[df.median().sort_values(ascending=False).head(20).index]
    mdf = df.melt()
    kwargs['ax'] = sns.boxplot(data=mdf, x=mdf.columns[0], y='value', showfliers=False, boxprops=dict(alpha=.25))
    sns.stripplot(data=mdf, x=mdf.columns[0], y='value', size=2, color=".3", ax=kwargs['ax'])
    kwargs['ax'].set_xlabel(mdf.columns[0])
    kwargs['ax'].set_ylabel('Relative abundance')
    plt.setp(kwargs['ax'].get_xticklabels(), rotation=45, ha="right")
    return kwargs['ax']

def box(**kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statannot
    try:
        stats = kwargs['stats']
        del kwargs['stats']
        stats = stats.loc[stats]
        if stats.sum() > 0:
            stats.loc[stats] = 0.05
    except:
        pass
    try: ax = kwargs['ax']
    except: fig, ax = plt.subplots(figsize=(4, 4))
    try:
        s = kwargs['s']
        del kwargs['s']
    except:
        pass
    sns.boxplot(showfliers=False, showcaps=False, **kwargs)
    try: del kwargs['palette']
    except: pass
    try:
        if kwargs['hue']:
            kwargs['dodge'] = True
    except: pass
    sns.stripplot(s=2, color="0.2", **kwargs)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    try:
        statannot.add_stat_annotation(
            ax,
            data=kwargs['data'],
            x=kwargs['x'],
            y=kwargs['y'],
            box_pairs=stats.index,
            perform_stat_test=False,
            pvalues=stats,
            text_format='star',
            verbose=0,
            )
    except: pass
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

def newcurve(df, mapping, ax=None):
    from scipy import stats
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None: fig, ax= plt.subplots(figsize=(4, 4))
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
    griddf.sort_index(axis=1).plot.area(stacked=True, color=mapping.to_dict(), ax=ax)
    plt.tick_params(bottom = False)
    #plt.xlim(0, 35)
    #plt.legend( title="Species", bbox_to_anchor=(1.001, 1), loc="upper left", fontsize="small")
    #plt.ylabel("Log(Relative abundance)")
    #plotdf = df.cumsum(axis=1).stack().reset_index()
    #sns.scatterplot(data=plotdf.sort_values(0), x='Days after birth', y=0, s=10, linewidth=0, ax=ax)
    return ax

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

def plot_network(edges):
    import networkx as nx
    G = nx.from_pandas_edgelist(edges.reset_index())
    ax = nx.draw(
            G,
            with_labels=True,
            node_color="b",
            node_size=50
            )
    return ax

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

def clusterplotnetwork(G):
    import networkx as nx
    ax = nx.draw(
            G,
            with_labels=True,
            node_color="b",
            node_size=50
            )
    return ax
    
def networkplot(G, group=None):
    import matplotlib.pyplot as plt
    from itertools import count
    import networkx as nx
    if group:
        nx.set_node_attributes(G, group, "group")
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
    
def dendrogram(df):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
    distance_matrix = np.array([[0, 1, 2, 3],
                                [1, 0, 4, 5],
                                [2, 4, 0, 6],
                                [3, 5, 6, 0]])
    Z = linkage(distance_matrix, method='average')
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    return ax

def polar(df):
    import numpy as np
    import matplotlib.pyplot as plt
    palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())
    ndf = df.loc[~df.index.str.contains('36'), df.columns.str.contains('Raw')].groupby(level=0).mean()
    data = ndf.T.copy().to_numpy()
    angles = np.linspace(0, 2*np.pi, len(ndf.columns), endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = ndf.columns.to_list()
    loopcategories = ndf.columns.to_list()
    loopcategories.append(df.columns[0])
    alldf = pd.DataFrame(data=data, index = loopcategories, columns=ndf.index).T
    allangles = pd.Series(data=angles, index=loopcategories)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for color in alldf.index.unique():
        plotdf = alldf.loc[alldf.index==color]
        ax.plot(allangles, plotdf.T, linewidth=1, color = palette[color])
    plt.title('Radial Line Graph')
    ax.set_xticks(allangles[:-1])
    ax.set_xticklabels(categories)
    ax.grid(True)

def sig(df, mult=False, perm=False):
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

def confusionmatrix(X, y, labels=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt
    import numpy as np
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    tprs, aucs = [],[]
    base_fpr = np.linspace(0, 1, 101)
    if not labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    return cm

def ANCOM(df, perm=False):
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

def corr(df1, df2, FDR=True, min_unique=10):
    import pandas as pd
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import fdrcorrection
    df1 = df1.loc[:,df1.nunique() > min_unique]
    df2 = df2.loc[:,df2.nunique() > min_unique]
    df = df1.join(df2, how='inner')
    cor, pval = spearmanr(df)
    cordf = pd.DataFrame(cor, index=df.columns, columns=df.columns)
    pvaldf = pd.DataFrame(pval, index=df.columns, columns=df.columns)
    cordf = cordf.loc[df1.columns, df2.columns]
    pvaldf = pvaldf.loc[df1.columns, df2.columns]
    pvaldf.fillna(1, inplace=True)
    if FDR:
        pvaldf = pd.DataFrame(
            fdrcorrection(pvaldf.values.flatten())[1].reshape(pvaldf.shape),
            index = pvaldf.index,
            columns = pvaldf.columns)
    return cordf, pvaldf

def filter(df, min_unique=0, thresh=None):
    import numpy as np
    df = df.loc[
            df.agg(np.count_nonzero, axis=1) > min_unique,
            df.agg(np.count_nonzero, axis=0) > min_unique]
    if thresh:
        df = df.loc[:, df.abs().gt(thresh).any(axis=0)]
    return df

def varianceexplained(df):
    import pandas as pd
    from skbio.stats.distance import permanova
    import numpy as np
    import skbio
    from scipy.spatial import distance
    df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    Ar_dist = distance.squareform(distance.pdist(df, metric="braycurtis"))
    DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
    result = permanova(DM_dist, df.index, permutations=10000)
    return result['p-value']
    #return result['test statistic']

def norm(df):
    return df.T.div(df.sum(axis=1), axis=1).T

def minmaxscale(df):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    scaledDf = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def StandardScale(df):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scaledDf = pd.DataFrame(
            StandardScaler().fit_transform(df),
            index=df.index,
            columns=df.columns)
    return scaledDf

def clr(df, axis=0):
    import pandas as pd
    from skbio.stats.composition import clr
    return pd.DataFrame(clr(df), index=df.index, columns=df.columns)

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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaledDf = StandardScaler().fit_transform(df)
    pca = PCA()
    results = pca.fit_transform(scaledDf).T
    df['PC1'], df['PC2'] = results[0,:], results[1,:]
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
    from scipy.stats import spearmanr
    import numpy as np
    import pandas as pd
    import shap
    explainer = shap.TreeExplainer(model)
    inter_shaps_values = explainer.shap_interaction_values(X)
    vals = inter_shaps_values[0]
    for i in range(1, vals.shape[0]):
        vals[0] += vals[i]
    final = pd.DataFrame(vals[0], index=X.columns, columns=X.columns)
    return final

def shannon(df, axis=1):
    from skbio.diversity.alpha import shannon
    return df.agg(shannon, axis=axis)

def evenness(df, axis=1):
    from skbio.diversity.alpha import pielou_e
    return df.agg(pielou_e, axis=axis)

def richness(df, axis=1):
    import numpy as np
    return df.agg(np.count_nonzero, axis=axis)

def beta(df):
    import pandas as pd
    from scipy.spatial import distance
    BC_dist = pd.DataFrame(
        distance.squareform(distance.pdist(df, metric="braycurtis")),
        columns=df.index,
        index=df.index) 
    return BC_dist

def to_edges(df, thresh=0.5, directional=True):
    import pandas as pd
    df = df.rename_axis('source', axis=0).rename_axis("target", axis=1)
    edges = df.stack().to_frame()[0]
    nedges = edges.reset_index()
    edges = nedges[nedges.target != nedges.source].set_index(['source','target']).drop_duplicates()[0]
    if directional:
        fin = edges.loc[(edges < 0.99) & (edges.abs() > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    else:
        fin = edges.loc[(edges < 0.99) & (edges > thresh)].dropna().reset_index().rename(columns={'level_0': 'source', 'level_1':'target', 0:'weight'}).set_index('source').sort_values('weight')
    return fin

def mult(df):
    import pandas as pd
    from skbio.stats.composition import multiplicative_replacement as mul
    return pd.DataFrame(mul(df), index=df.index, columns=df.columns)

def taxofunc(msp, taxo, short=False):
    import pandas as pd
    m, t = msp.copy(), taxo.copy()
    t['superkingdom'] = 'k_' + t['superkingdom']
    t['phylum'] = t[['superkingdom', 'phylum']].apply(lambda x: '|p_'.join(x.dropna().astype(str).values), axis=1)
    t['class'] = t[['phylum', 'class']].apply(lambda x: '|c_'.join(x.dropna().astype(str).values), axis=1)
    t['order'] = t[['class', 'order']].apply(lambda x: '|o_'.join(x.dropna().astype(str).values), axis=1)
    t['family'] = t[['order', 'family']].apply(lambda x: '|f_'.join(x.dropna().astype(str).values), axis=1)
    t['genus'] = t[['family', 'genus']].apply(lambda x: '|g_'.join(x.dropna().astype(str).values), axis=1)
    t['species'] = t[['genus', 'species']].apply(lambda x: '|s_'.join(x.dropna().astype(str).values), axis=1)
    mt = m.join(t, how='inner')
    df = pd.concat([mt.set_index(t.columns[i])._get_numeric_data().groupby(level=0).sum() for i in range(len(t.columns))])
    df.index = df.index.str.replace(" ", "_", regex=True).str.replace('/.*','', regex=True)
    if short:
        df.index = df.T.add_prefix("|").T.index.str.extract(".*\|([a-z]_.*$)", expand=True)[0]
    df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    return df

def cluster(G):
    from networkx.algorithms.community.centrality import girvan_newman as cluster
    import pandas as pd
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

def convtoRGBA(df):
    import seaborn as sns
    import pandas as pd
    for col in df.columns:
        lut = pd.Series(
                sns.color_palette("hls", df[col].sort_values().nunique()).as_hex(),
                index=df[col].sort_values().unique())
        df[col] = df[col].map(lut)
    return df

### Workflows
def diversityanalysis(df, subject):
    import pandas as pd
    import numpy as np
    diversity = pd.concat(
            [f.evenness(df),
             f.richness(df),
             f.shannon(df)],
            axis=1).sort_index().sort_index(ascending=False)
    diversity.droplevel(1).to_csv(f'../results/{subject}Diversity.csv')
    diversity = diversity.droplevel(0)
    fc = f.lfc(diversity)
    fc.columns = fc.columns.str.join('/')
    fc.T.to_csv(f'../results/{subject}FCdiversity.csv')
    sig = f.sig(diversity)
    sig.columns = sig.columns.str.join('/')
    sig.T.to_csv(f'../results/{subject}MWWdiversity.csv')
    pcoa = f.PCOA(df.droplevel(0).sort_index())
    pcoa.to_csv(f'../results/{subject}PCOA.csv')
    permanova = f.PERMANOVA(df.reset_index(drop=True), df.index)
    with open(f'../results/{subject}permanova.txt','w') as of: of.write(permanova.to_string())
    comb = list(itertools.combinations(df.index.unique(), 2))
    out= pd.DataFrame()
    i = comb[0]
    for i in comb: 
        out[i] = f.PERMANOVA(df.loc[list(i)].reset_index(drop=True), df.loc[list(i)].index)
    out.to_csv(f'../results/{subject}pairwiseANOVA.csv')

def differentialAbundance(df, subject):
    import numpy as np 
    fc = lfc(df, perm=True)
    fc.columns = fc.columns.str.join('/')
    fc = fc.replace([np.inf, -np.inf], np.nan).dropna()
    pval = sig(df, mult=True, perm=True)
    #pval = ANCOM(df, perm=True).T
    pval.columns = pval.columns.str.join('/')
    pval = pval.loc[fc.index]
    fc.to_csv(f'../results/{subject}FCabundance.csv')
    pval.to_csv(f'../results/{subject}MWWabundance.csv')

def differentialAbundance_plot(subject):
    fc = pd.read_csv('../results/{subject}FCabundance.csv', index_col=0)
    sig = pd.read_csv('../results/{subject}MWWabundance.csv', index_col=0)
    fc.columns = fc.columns.str.split('/', expand=True)
    sig.columns = sig.columns.str.split('/', expand=True)
    val = 0.05
    fcthresh = 1
    hfc = fc.xs('Healthy', axis=1, level=1)
    hsig = sig.xs('Healthy', axis=1, level=1)
    ffc = hfc.loc[hfc.abs().gt(fcthresh).any(axis=1) & hsig.lt(val).any(axis=1)]
    fsig = hsig.loc[hfc.abs().gt(fcthresh).any(axis=1) & hsig.lt(val).any(axis=1)]
    f.clustermap(ffc, fsig.lt(val), figsize=(3,7))
    plt.savefig(f'../results/{subject}ClusterFC_HvD.svg')

def differentialAbundance_vennplot(var1, var2, var3, subject, val=0.05, fcthresh=1):
    d1 = hfc[var1].loc[hsig.lt(val) & hfc.lt(-fcthresh).AML].index
    d2 = hfc.MDS.loc[hsig.lt(val).MDS & hfc.lt(-fcthresh).MDS].index
    d3 = hfc.MPN.loc[hsig.lt(val).MPN & ffc.lt(-fcthresh).MPN].index
    fig, ax = plt.subplots(figsize=(2,2))
    venn3(subsets=[set(d1),set(d2),set(d3)], set_labels=[var1,var2,var3], ax=ax)
    plt.savefig(f'../results/{subject}VennDep.svg')
    uaml = hfc.AML.loc[hsig.lt(val).AML & hfc.gt(fcthresh).AML].index
    umds = hfc.MDS.loc[hsig.lt(val).MDS & hfc.gt(fcthresh).MDS].index
    umpn = hfc.MPN.loc[hsig.lt(val).MPN & hfc.gt(fcthresh).MPN].index
    fig, ax = plt.subplots(figsize=(2,2))
    venn3(subsets=[set(umds),set(uaml),set(umpn)], set_labels=['MDS','AML','MPN'], set_colors=colours, ax=ax)
    plt.savefig(f'../results/{subject}mspVennEnrich.svg')

def fbratio(metamsp, taxo):
    var='phylum'
    metaPhylumMsp = metamsp.T.join(taxo[var]).groupby(var).sum().T
    metaPhylumMsp = metaPhylumMsp.loc[metaPhylumMsp.sum(axis=1) != 0, metaPhylumMsp.sum(axis=0) !=0]
    FB = metaPhylumMsp.Firmicutes.div(metaPhylumMsp.Bacteroidota)
    FB.replace([np.inf, -np.inf], np.nan, inplace=True)
    FB.dropna(inplace=True)
    FB = FB.reset_index().set_axis(['Host Phenotype', 'F/B Ratio'], axis=1).set_index('Host Phenotype')
    FB = FB.sort_index()
    FB.to_csv('../results/FBratio.csv')
    lfc = f.lfc(FB).T.set_axis(['F/B_FC'], axis=1)
    sig = f.sig(FB).T.set_axis(['F/B_M.W.W'], axis=1)
    joined = lfc.join(sig)
    joined.index = joined.index.str.join('/')
    joined.to_csv('../results/FBratioFC.csv')

def fbratio_plot():
    FB = pd.read_csv('../results/FBratio.csv', index_col=0).sort_index()
    stats = pd.read_csv('../results/FBratioFC.csv', index_col=0)
    stats.index = stats.index.str.split('/')
    fig, ax = plt.subplots(figsize=(1.5,1.5))
    f.box(data=FB, x=FB.index, y='F/B Ratio', palette=colours.to_dict(), stats=stats['F/B_M.W.W'].lt(0.05), ax=ax)
    ax.set_yscale('log')
    plt.legend([],[], frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.ylim([0.1,1000])
    plt.savefig(f'../results/FBratiobox.svg')

def phylumRelabundance(metamsp, taxo):
    phy = metamsp.T.join(taxo['phylum']).groupby('phylum').sum().T
    ndf = phy.groupby(phy.index).mean()
    nndf = ndf[ndf.sum().sort_values().tail(8).index]
    nndf.to_csv('../results/phylumRelabund.csv')

if __name__ == '__main__':
    import doctest
    doctest.testmod()
