[1mdiff --git a/metatoolkit/describe.py b/metatoolkit/describe.py[m
[1mindex addd4ce..8c4099c 100644[m
[1m--- a/metatoolkit/describe.py[m
[1m+++ b/metatoolkit/describe.py[m
[36m@@ -15,7 +15,7 @@[m [mparser.add_argument('-r', '--corr', action='store_true')[m
 known = parser.parse_args()[m
 known = {k: v for k, v in vars(known).items() if v is not None}[m
 [m
[31m-def describe(df, datatype=None, **kwargs):[m
[32m+[m[32mdef Describe(df, datatype=None, **kwargs):[m
     available = {'change_summary':change_summary,[m
                  'corr_summary':corr_summary,[m
                  'mbiome_summary':mbiome_summary}[m
[36m@@ -75,6 +75,6 @@[m [mif os.path.isfile(subject): subject = Path(subject).stem[m
 df = f.load(subject)[m
 [m
 df = f.load(subject)[m
[31m-output = describe(df, **known)[m
[32m+[m[32moutput = Describe(df, **known)[m
 print(output.to_string())[m
[31m-f.save(output, f'{subject}describe')[m
[32m+[m[32mf.save(output, f'{subject}Describe')[m
[1mdiff --git a/metatoolkit/functions.py b/metatoolkit/functions.py[m
[1mindex 21282fa..f290dbe 100755[m
[1m--- a/metatoolkit/functions.py[m
[1m+++ b/metatoolkit/functions.py[m
[36m@@ -574,11 +574,8 @@[m [mdef merge(datasets=None, join='inner', append=None):[m
         outdf = pd.concat(datasets, axis=1, join=join)[m
     return outdf[m
 [m
[31m-def group(df, type='sum'):[m
[31m-    if type=='sum':[m
[31m-        outdf = df.groupby(level=0).sum()[m
[31m-    elif type=='mean':[m
[31m-        outdf = df.groupby(level=0).mean()[m
[32m+[m[32mdef group(df, agg_func='sum'):[m
[32m+[m[32m    outdf = df.groupby(level=0).agg(agg_func)[m
     return outdf[m
 [m
 def filter(df, **kwargs):[m
[36m@@ -827,7 +824,7 @@[m [mdef change(df, df2, columns=None, analysis=['mww','fc','diffmean','summary']):[m
     if columns is None: columns = df2.columns[m
     available={'mww':mww, 'fc':fc, 'diffmean':diffmean, 'summary':summary}[m
     i = 'summary'[m
[31m-    label = df2.columns[2][m
[32m+[m[32m    #label = df2.columns[2][m
     out = {}[m
     for label in df2[columns].columns:[m
         output = [][m
[36m@@ -945,7 +942,7 @@[m [mdef prevail(df):[m
     return df.agg(np.count_nonzero, axis=0).div(df.shape[0]).to_frame('baseprev')[m
 [m
 def onehot(df):[m
[31m-    return pd.get_dummies(df, prefix_sep='.')[m
[32m+[m[32m    return pd.get_dummies(df, prefix_sep='.', dtype=bool)[m
 [m
 def calculate(analysis, df):[m
     available={[m
[36m@@ -1113,7 +1110,7 @@[m [mdef savefig(subject, tl=False, show=False):[m
     if tl: plt.tight_layout()[m
     plt.savefig(f'../results/{subject}.svg')[m
     plt.savefig(f'../results/{subject}.pdf')[m
[31m-    plt.savefig(f'../results/{subject}.jpg')[m
[32m+[m[32m    plt.savefig(f'../results/{subject}.jpg', dpi=500)[m
     if show: plt.show()[m
     subprocess.call(f'zathura ../results/{subject}.pdf &', shell=True)[m
     plt.clf()[m
[1mdiff --git a/metatoolkit/makesupptable.py b/metatoolkit/makesupptable.py[m
[1mindex b8024ce..d3bd02a 100644[m
[1m--- a/metatoolkit/makesupptable.py[m
[1m+++ b/metatoolkit/makesupptable.py[m
[36m@@ -1,31 +1,60 @@[m
 #!/usr/bin/env python[m
 import os[m
[31m-import pathlib[m
 import pandas as pd[m
[31m-import shutil[m
[31m-import subprocess[m
[31m-import sys[m
[31m-[m
[31m-# Create Supp tables in excel[m
[31m-tables = pd.read_csv('../figures/suppTableList.txt', header=None)[m
[31m-writer = pd.ExcelWriter('../figures/suppTables.xlsx')[m
[31m-table = '../results/' + tables + '.tsv'[m
[32m+[m[32mfrom openpyxl import load_workbook[m
[32m+[m[32mfrom openpyxl.utils import get_column_letter[m
[32m+[m[32mfrom openpyxl.styles import Font, Alignment, Border, Side[m
[32m+[m
[32m+[m[32m# Paths[m
[32m+[m[32msupp_table_list_path = '../figures/suppTableList.txt'[m
[32m+[m[32mcontents_path = '../figures/contents.tsv'[m
[32m+[m[32mglossary_path = '../figures/glossary.tsv'[m
[32m+[m[32moutput_path = '../figures/suppTables.xlsx'[m
[32m+[m
[32m+[m[32m# Read the list of tables to be included[m
[32m+[m[32mtables = pd.read_csv(supp_table_list_path, header=None)[m
[32m+[m[32mtable_paths = '../results/' + tables[0] + '.tsv'[m
[32m+[m
 # Add Glossary[m
[31m-table = pd.concat([pd.Series('../figures/Supp_table_glossary.tsv'), table])[m
[31m-with pd.ExcelWriter('../figures/suppTables.xlsx') as writer:[m
[31m-    for j,i in enumerate(table[0]):[m
[31m-        pd.read_csv(i, sep='\t', index_col=0).to_excel([m
[31m-                writer,[m
[31m-                sheet_name='SuppT' + str(j) + '_' + i.split('/')[-1].split('.')[0])[m
[31m-[m
[31m-'''[m
[31m-# Create Supp tables for latex[m
[31m-tables = pd.read_csv('../figures/suppTableList.txt', header=None)[m
[31m-table = '../results/' + tables + '.tsv'[m
[31m-mydir='../figures/texsupptables'[m
[31m-try: shutil.rmtree(mydir)[m
[31m-except: None[m
[31m-pathlib.Path(mydir).mkdir(parents=True) [m
[31m-for j,i in enumerate(table[0]):[m
[31m-    pd.read_csv(i, sep='\t', index_col=0, dtype=object).to_latex('../figures/texsupptables/TableS' + str(j+1) + '_' + i.split('/')[-1].split('.')[0] + '.tex', escape="Latex", longtable=True)[m
[31m-'''[m
[32m+[m[32mtable_paths = pd.concat([pd.Series([contents_path, glossary_path]), table_paths])[m
[32m+[m
[32m+[m[32m# Write tables to Excel file[m
[32m+[m[32mwith pd.ExcelWriter(output_path, engine='openpyxl') as writer:[m
[32m+[m[32m    for j, table_path in enumerate(table_paths):[m
[32m+[m[32m        # Read each table and export to the Excel file[m
[32m+[m[32m        df = pd.read_csv(table_path, sep='\t', index_col=0)[m
[32m+[m[32m        sheet_name = f"Data_{j-1}"[m
[32m+[m[32m        if j == 0:[m
[32m+[m[32m            sheet_name = 'Contents'[m
[32m+[m[32m        if j == 1:[m
[32m+[m[32m            sheet_name = 'Glossary'[m
[32m+[m[32m        df.to_excel(writer, sheet_name=sheet_name)[m
[32m+[m
[32m+[m[32m# Define border styles[m
[32m+[m[32mthin_border = Border(bottom=Side(style="thin"))[m
[32m+[m
[32m+[m[32m# Apply formatting[m
[32m+[m[32mworkbook = load_workbook(output_path)[m
[32m+[m[32mfor sheet_name in workbook.sheetnames:[m
[32m+[m[32m    sheet = workbook[sheet_name][m
[32m+[m
[32m+[m[32m    # Clear formatting: remove bold in the first column and all borders[m
[32m+[m[32m    for row in sheet.iter_rows():[m
[32m+[m[32m        for cell in row:[m
[32m+[m[32m            cell.font = Font(bold=False)  # Remove bold from all cells[m
[32m+[m[32m            cell.border = Border()  # Remove all borders[m
[32m+[m
[32m+[m[32m    # Apply header formatting (bold, center, bottom border)[m
[32m+[m[32m    for cell in sheet[1]:  # Assumes first row is the header[m
[32m+[m[32m        cell.font = Font(bold=True)[m
[32m+[m[32m        cell.alignment = Alignment(horizontal='center')[m
[32m+[m[32m        cell.border = thin_border  # Apply bottom border only to header[m
[32m+[m
[32m+[m[32m    # Set column width based on content[m
[32m+[m[32m    for col in sheet.columns:[m
[32m+[m[32m        max_length = max(len(str(cell.value)) for cell in col if cell.value) + 2[m
[32m+[m[32m        col_letter = get_column_letter(col[0].column)[m
[32m+[m[32m        sheet.column_dimensions[col_letter].width = max(max_length, 12)[m
[32m+[m
[32m+[m[32m# Save formatted workbook[m
[32m+[m[32mworkbook.save(output_path)[m
[1mdiff --git a/metatoolkit/merge.py b/metatoolkit/merge.py[m
[1mindex 3fce222..7edb8f6 100755[m
[1m--- a/metatoolkit/merge.py[m
[1m+++ b/metatoolkit/merge.py[m
[36m@@ -12,33 +12,6 @@[m [mparser.add_argument('-o', '--output')[m
 known = parser.parse_args()[m
 known = {k: v for k, v in vars(known).items() if v is not None}[m
 [m
[31m-'''[m
[31m-dfs = [[m
[31m-        'aaID',[m
[31m-        'anthroID',[m
[31m-        'bayleyID',[m
[31m-        'classID',[m
[31m-        'familyID',[m
[31m-        'fnirsID',[m
[31m-        'geneticsID',[m
[31m-        'genusID',[m
[31m-        'kingdomID',[m
[31m-        'lipidsID',[m
[31m-        'microID',[m
[31m-        'orderID',[m
[31m-        'pathwaysallID',[m
[31m-        'pathwaysID',[m
[31m-        'pathwaystaxoID',[m
[31m-        'phylumID',[m
[31m-        'psdID',[m
[31m-        'sleepID',[m
[31m-        'speciesID',[m
[31m-        'taxoID',[m
[31m-        'vepID',[m
[31m-        'wolkesID'[m
[31m-        ][m
[31m-'''[m
[31m-[m
 dfs = known.get("datasets")[m
 alldfs = [f.load(df) for df in dfs][m
 join=known.get("join") if known.get("join") else 'inner'[m
[1mdiff --git a/metatoolkit/networkplot.py b/metatoolkit/networkplot.py[m
[1mdeleted file mode 100755[m
[1mindex a016d52..0000000[m
[1m--- a/metatoolkit/networkplot.py[m
[1m+++ /dev/null[m
[36m@@ -1,38 +0,0 @@[m
[31m-#!/usr/bin/env python[m
[31m-[m
[31m-import argparse[m
[31m-import networkx as nx[m
[31m-import matplotlib.pyplot as plt[m
[31m-import pandas as pd[m
[31m-[m
[31m-def plot_network(edge_list, node_list=None):[m
[31m-    # Create an empty graph[m
[31m-    G = nx.Graph()[m
[31m-[m
[31m-    # Read edge list and add edges to the graph[m
[31m-    edge_df = pd.read_csv(edge_list, sep='\t')[m
[31m-    edges = [(str(row['source']), str(row['target'])) for _, row in edge_df.iterrows()][m
[31m-    G.add_edges_from(edges)[m
[31m-[m
[31m-    # If node list is provided, read it and add node colors[m
[31m-    if node_list:[m
[31m-        node_df = pd.read_csv(node_list, sep='\t')[m
[31m-        node_colors = {str(row['node']): row['cluster'] for _, row in node_df.iterrows()}[m
[31m-        # Draw nodes with specified colors[m
[31m-        node_color = [node_colors.get(node, 'blue') for node in G.nodes()][m
[31m-    else:[m
[31m-        node_color = 'blue'[m
[31m-[m
[31m-    # Plot the network[m
[31m-    nx.draw(G, with_labels=True, node_color=node_color)[m
[31m-    plt.show()[m
[31m-[m
[31m-if __name__ == "__main__":[m
[31m-    # Parse command line arguments[m
[31m-    parser = argparse.ArgumentParser(description='Plot a network graph.')[m
[31m-    parser.add_argument('--edge-list', required=True, help='Path to the edge list TSV file')[m
[31m-    parser.add_argument('--node-list', help='Path to the node list TSV file with colors (optional)')[m
[31m-    args = parser.parse_args()[m
[31m-[m
[31m-    # Plot the network[m
[31m-    plot_network(args.edge_list, args.node_list)[m
[1mdiff --git a/metatoolkit/pcoa.py b/metatoolkit/pcoa.py[m
[1mindex ead701d..cc2f7c2 100644[m
[1m--- a/metatoolkit/pcoa.py[m
[1m+++ b/metatoolkit/pcoa.py[m
[36m@@ -1,39 +1,85 @@[m
 #!/usr/bin/env python[m
 # -*- coding: utf-8 -*-[m
[31m-[m
 import argparse[m
[31m-import functions as f[m
[31m-import pandas as pd[m
[31m-from pathlib import Path[m
 import os[m
[32m+[m[32mimport pandas as pd[m
 import skbio[m
 [m
[31m-parser = argparse.ArgumentParser(description='Filter')[m
[31m-parser.add_argument('subject', type=str, help='Distance matrix')[m
[31m-parser.add_argument('-o', '--outfile', type=str)[m
[31m-parser.add_argument('-s', '--suffix', type=str)[m
[31m-known = parser.parse_args()[m
[31m-known = {k: v for k, v in vars(known).items() if v is not None}[m
[31m-[m
[31m-subject = known.get("subject"); known.pop('subject')[m
[31m-if os.path.isfile(subject): subject = Path(subject).stem[m
[31m-df = f.load(subject)[m
[31m-[m
[31m-outfile = known.get("outfile") if known.get("outfile") else None[m
[31m-suffix = known.get("suffix") if known.get("suffix") else None[m
[31m-[m
[31m-output = pd.DataFrame(index=df.index)[m
[31m-DM_dist = skbio.stats.distance.DistanceMatrix(df)[m
[31m-PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)[m
[31m-label = PCoA.proportion_explained.apply(' ({:.1%})'.format)[m
[31m-results = PCoA.samples.copy()[m
[31m-output['PCo1' + label.loc['PC1']], output['PCo2' + label.loc['PC2']] = results.iloc[:,0].values, results.iloc[:,1].values[m
[31m-[m
[31m-print(output)[m
[31m-if output is not None:[m
[31m-    if outfile:[m
[31m-        f.save(output, outfile)[m
[31m-    elif suffix:[m
[31m-        f.save(output, subject + outfile)[m
[32m+[m[32mdef parse_arguments():[m
[32m+[m[32m    """Parse command-line arguments."""[m
[32m+[m[32m    parser = argparse.ArgumentParser(description="Perform PCoA on a distance matrix")[m
[32m+[m[32m    parser.add_argument("-i", "--input", type=str, help="Distance matrix path")[m
[32m+[m[32m    parser.add_argument("-o", "--output", type=str, help="Output file path")[m
[32m+[m[32m    return parser.parse_args()[m
[32m+[m
[32m+[m[32mdef perform_pcoa(df):[m
[32m+[m[32m    """[m
[32m+[m[32m    Perform PCoA on the given distance matrix DataFrame.[m
[32m+[m[32m    Parameters:[m
[32m+[m[32m    df (pd.DataFrame): A square DataFrame representing a distance matrix, where[m
[32m+[m[32m                       rows and columns are samples, and values are distances.[m
[32m+[m[32m    Returns:[m
[32m+[m[32m    pd.DataFrame: A DataFrame with the first two PCoA components (PCo1 and PCo2) for each sample.[m
[32m+[m[32m    Example:[m
[32m+[m[32m    --------[m
[32m+[m[32m    Test data input:[m
[32m+[m[32m    Consider the following sample distance matrix:[m
[32m+[m[32m        A      B      C[m
[32m+[m[32m    A  0.0    0.1    0.3[m
[32m+[m[32m    B  0.1    0.0    0.4[m
[32m+[m[32m    C  0.3    0.4    0.0[m
[32m+[m[32m    Create the DataFrame and run PCoA:[m
[32m+[m[32m    >>> import pandas as pd[m
[32m+[m[32m    >>> import skbio[m
[32m+[m[32m    >>> data = {'A': [0.0, 0.1, 0.3], 'B': [0.1, 0.0, 0.4], 'C': [0.3, 0.4, 0.0]}[m
[32m+[m[32m    >>> df = pd.DataFrame(data, index=['A', 'B', 'C'])[m
[32m+[m[32m    >>> perform_pcoa(df)[m
[32m+[m[32m       PCo1 (100.0%)  PCo2 (0.0%)[m
[32m+[m[32m    A      -0.066667          0.0[m
[32m+[m[32m    B      -0.166667         -0.0[m
[32m+[m[32m    C       0.233333          0.0[m
[32m+[m[32m    """[m
[32m+[m[32m    DM_dist = skbio.stats.distance.DistanceMatrix(df)[m
[32m+[m[32m    PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=2)[m
[32m+[m[32m    label = PCoA.proportion_explained.apply(" ({:.1%})".format)[m
[32m+[m[32m    results = PCoA.samples.copy()[m
[32m+[m[32m    result = pd.DataFrame(index=df.index)[m
[32m+[m[32m    result["PCo1" + label.loc["PC1"]] = results.iloc[:, 0].values[m
[32m+[m[32m    result["PCo2" + label.loc["PC2"]] = results.iloc[:, 1].values[m
[32m+[m[32m    return result[m
[32m+[m
[32m+[m[32mdef main():[m
[32m+[m[32m    """Main function to execute the script."""[m
[32m+[m[32m    args = parse_arguments()[m
[32m+[m[32m    subject = args.input[m
[32m+[m
[32m+[m[32m    # Read input data[m
[32m+[m[32m    try:[m
[32m+[m[32m        df = pd.read_csv(subject, sep='\t', index_col=0)[m
[32m+[m[32m    except Exception as e:[m
[32m+[m[32m        print(f"Error reading input file {subject}: {e}")[m
[32m+[m[32m        return[m
[32m+[m
[32m+[m[32m    # Perform PCoA[m
[32m+[m[32m    result = perform_pcoa(df)[m
[32m+[m
[32m+[m[32m    # Determine output path[m
[32m+[m[32m    if args.output:[m
[32m+[m[32m        output_path = args.output[m
     else:[m
[31m-        f.save(output, subject + 'Pcoa')[m
[32m+[m[32m        # Default output path[m
[32m+[m[32m        base_name = os.path.splitext(os.path.basename(subject))[0][m
[32m+[m[32m        output_path = f'../results/{base_name}_pcoa.tsv'[m
[32m+[m
[32m+[m[32m    # Ensure the directory exists[m
[32m+[m[32m    os.makedirs(os.path.dirname(output_path), exist_ok=True)[m
[32m+[m
[32m+[m[32m    # Save results to file[m
[32m+[m[32m    try:[m
[32m+[m[32m        result.to_csv(output_path, sep='\t')[m
[32m+[m[32m        print(f"PCoA results saved to: {output_path}")[m
[32m+[m[32m    except Exception as e:[m
[32m+[m[32m        print(f"Error saving results to {output_path}: {e}")[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    main()[m
[1mdiff --git a/metatoolkit/regplot.py b/metatoolkit/regplot.py[m
[1mindex 61d586d..e09009b 100755[m
[1m--- a/metatoolkit/regplot.py[m
[1m+++ b/metatoolkit/regplot.py[m
[36m@@ -1,6 +1,7 @@[m
 #!/usr/bin/env python[m
 # -*- coding: utf-8 -*-[m
 [m
[32m+[m[32mimport seaborn as sns[m
 import argparse[m
 import functions as f[m
 import matplotlib.pyplot as plt[m
[36m@@ -22,9 +23,17 @@[m [mknown = {k: v for k, v in vars(known).items() if v is not None}[m
 subject = known.get("subject"); known.pop("subject")[m
 df = f.load(subject)[m
 logy = known.get("logy"); known.pop("logy")[m
[32m+[m[32mhue = known.get('hue')[m
 [m
 f.setupplot()[m
[31m-f.regplot(df, **known)[m
[32m+[m[32mif hue:[m
[32m+[m[32m    x = known.get('x')[m
[32m+[m[32m    y = known.get('y')[m
[32m+[m[32m    sns.regplot(data = df, x=x, y=y, color='red', scatter=False)[m
[32m+[m[32m    sns.scatterplot(data = df, x=x, y=y, s=2, hue=hue, legend=False)[m
[32m+[m[32melse:[m
[32m+[m[32m    f.regplot(df, **known)[m
[32m+[m
 if logy: plt.yscale('log')[m
 f.savefig(f'{subject}regplot')[m
 [m
[1mdiff --git a/metatoolkit/sig_summary.py b/metatoolkit/sig_summary.py[m
[1mindex d023431..c115fa9 100644[m
[1m--- a/metatoolkit/sig_summary.py[m
[1m+++ b/metatoolkit/sig_summary.py[m
[36m@@ -7,14 +7,6 @@[m [mimport pandas as pd[m
 import os[m
 from pathlib import Path[m
 [m
[31m-parser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')[m
[31m-parser.add_argument('subject')[m
[31m-parser.add_argument('-p', '--pval', type=float, default=0.25)[m
[31m-parser.add_argument('-c', '--change', type=str, default='coef')[m
[31m-parser.add_argument('-s', '--sig', type=str, default='qval')[m
[31m-known = parser.parse_args()[m
[31m-known = {k: v for k, v in vars(known).items() if v is not None}[m
[31m-[m
 def change_summary(df, change='coef', sig='qval', pval=0.25):[m
     total_rows = df.shape[0][m
     sig_changed_count = df[sig].lt(pval).sum()[m
[36m@@ -25,6 +17,14 @@[m [mdef change_summary(df, change='coef', sig='qval', pval=0.25):[m
     decreased = f"sig down = {sig_decreased_count}/{total_rows} ({round(sig_decreased_count / total_rows * 100)}%)"[m
     return pd.Series([changed, increased, decreased])[m
 [m
[32m+[m[32mparser = argparse.ArgumentParser(description='Describe - Produces a summary report of analysis')[m
[32m+[m[32mparser.add_argument('subject')[m
[32m+[m[32mparser.add_argument('-p', '--pval', type=float, default=0.25)[m
[32m+[m[32mparser.add_argument('-c', '--change', type=str, default='coef')[m
[32m+[m[32mparser.add_argument('-s', '--sig', type=str, default='qval')[m
[32m+[m[32mknown = parser.parse_args()[m
[32m+[m[32mknown = {k: v for k, v in vars(known).items() if v is not None}[m
[32m+[m
 subject = known.get("subject")[m
 if os.path.isfile(subject): subject = Path(subject).stem[m
 df = f.load(subject)[m
[1mdiff --git a/metatoolkit/spindle.py b/metatoolkit/spindle.py[m
[1mindex 530fe4a..9e98797 100755[m
[1m--- a/metatoolkit/spindle.py[m
[1m+++ b/metatoolkit/spindle.py[m
[36m@@ -7,9 +7,38 @@[m [mimport matplotlib.pyplot as plt[m
 import pandas as pd[m
 import os[m
 from pathlib import Path[m
[32m+[m[32mimport seaborn as sns[m
[32m+[m
[32m+[m[32mdef spindle(df, meta=None, ax=None, palette=None, **kwargs):[m
[32m+[m[32m    if palette is None: palette = pd.Series(sns.color_palette("hls", df.index.nunique()).as_hex(), index=df.index.unique())[m
[32m+[m[32m    if ax is None: fig, ax= plt.subplots()[m
[32m+[m[32m    x=df.columns[0][m
[32m+[m[32m    y=df.columns[1][m
[32m+[m[32m    centers = df.groupby(df.index).mean()[m
[32m+[m[32m    centers.columns=['nPC1','nPC2'][m
[32m+[m[32m    j = df.join(centers)[m
[32m+[m[32m    j['colours'] = palette[m
[32m+[m[32m    i = j.reset_index().index[0][m
[32m+[m[32m    for i in j.reset_index().index:[m
[32m+[m[32m        ax.plot([m
[32m+[m[32m            j[[x,'nPC1']].iloc[i],[m
[32m+[m[32m            j[[y,'nPC2']].iloc[i],[m
[32m+[m[32m            linewidth = 0.5,[m
[32m+[m[32m            color = j['colours'].iloc[i],[m
[32m+[m[32m            zorder=1,[m
[32m+[m[32m            alpha=0.3[m
[32m+[m[32m        )[m
[32m+[m[32m        ax.scatter(j[x].iloc[i], j[y].iloc[i], color = j['colours'].iloc[i], s=1)[m
[32m+[m[32m    for i in centers.index:[m
[32m+[m[32m        ax.text(centers.loc[i,'nPC1']+0.002,centers.loc[i,'nPC2']+0.002, s=i, zorder=3)[m
[32m+[m[32m    ax.scatter(centers.nPC1, centers.nPC2, c='black', zorder=2, s=10, marker='+')[m
[32m+[m[32m    ax.set_xlabel(x)[m
[32m+[m[32m    ax.set_ylabel(y)[m
[32m+[m[32m    ax.spines[['right', 'top']].set_visible(False)[m
[32m+[m[32m    return ax[m
 [m
 parser = argparse.ArgumentParser(description='''[m
[31m-Heatmap - Produces a heatmap of a given dataset[m
[32m+[m[32mSpindle - Produces a spindleplot of a given dataset[m
 ''')[m
 parser.add_argument('subject')[m
 known = parser.parse_args()[m
[36m@@ -20,6 +49,38 @@[m [mif os.path.isfile(subject): subject = Path(subject).stem[m
 df = f.load(subject)[m
 [m
 f.setupplot()[m
[31m-f.spindle(df)[m
[32m+[m[32mspindle(df)[m
 f.savefig(f'{subject}Spindle')[m
 [m
[32m+[m
[32m+[m[32m'''[m
[32m+[m[32mdef parse_args(args):[m
[32m+[m[32m    parser = argparse.ArgumentParser([m
[32m+[m[32m       prog='predict.py',[m
[32m+[m[32m       description='Random Forest Classifier/Regressor with options'[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument('analysis', type=str, help='Regressor or Classifier')[m
[32m+[m[32m    parser.add_argument('subject', type=str, help='Data name or full filepath')[m
[32m+[m[32m    parser.add_argument('-n','--n_iter', type=int, help='Number of iterations for bootstrapping', default=10)[m
[32m+[m[32m    parser.add_argument('--shap_val', action='store_true', help='SHAP interpreted output')[m
[32m+[m[32m    parser.add_argument('--shap_interact', action='store_true', help='SHAP interaction interpreted output')[m
[32m+[m[32m    return parser.parse_args(args)[m
[32m+[m
[32m+[m[32m#arguments = ['classifier','speciesCondition.MAM','--shap_val','--shap_interact', '-n=20'][m
[32m+[m[32marguments = sys.argv[1:][m
[32m+[m[32margs = parse_args(arguments)[m
[32m+[m
[32m+[m[32m# Check if the provided subject is a valid file path[m
[32m+[m[32mif os.path.isfile(args.subject):[m
[32m+[m[32m    subject = Path(args.subject).stem[m
[32m+[m[32melse:[m
[32m+[m[32m    subject = args.subject[m
[32m+[m
[32m+[m[32mdf = f.load(subject)[m
[32m+[m[32manalysis = args.analysis[m
[32m+[m[32mshap_val = args.shap_val[m
[32m+[m[32mshap_interact = args.shap_interact[m
[32m+[m[32mn_iter = args.n_iter[m
[32m+[m
[32m+[m[32mpredict(df, args.analysis, shap_val=args.shap_val, shap_interact=args.shap_interact, n_iter=args.n_iter)[m
[32m+[m[32m'''[m
[1mdiff --git a/metatoolkit/volcano.py b/metatoolkit/volcano.py[m
[1mindex df4444b..627d476 100755[m
[1m--- a/metatoolkit/volcano.py[m
[1m+++ b/metatoolkit/volcano.py[m
[36m@@ -25,8 +25,8 @@[m [munknown = eval(unknown[0]) if unknown != [] else {}[m
 [m
 # Assemble params[m
 subject = known.get("subject"); known.pop('subject')[m
[31m-change = known.get("change") if known.get("change") else 'Log2FC'[m
[31m-sig = known.get("sig") if known.get("sig") else 'MWW_pval'[m
[32m+[m[32mchange = known.get("change") if known.get("change") else 'coef'[m
[32m+[m[32msig = known.get("sig") if known.get("sig") else 'qval'[m
 fc = float(known.get("fc")) if known.get("fc") else 1.0[m
 pval = float(known.get("pval")) if known.get("pval") else 0.05[m
 annot = known.get("annot") if known.get("annot") else True[m
