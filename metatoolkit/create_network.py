#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import networkx as nx

def load_nodes(nodes_file):
    """Load nodes from a TSV file, including all attributes."""
    nodes_df = pd.read_csv(nodes_file, sep='\t')
    if nodes_df.shape[1] < 1:
        raise ValueError("The nodes file must have at least one column for node identifiers.")
    
    # Use the first column as the node identifier
    nodes_df = nodes_df.set_index(nodes_df.columns[0])
    nodes_dict = nodes_df.to_dict(orient='index')
    return nodes_dict

def load_edges(edges_file, source_col='source', target_col='target'):
    """Load edges from a TSV file, allowing custom source and target column names."""
    edges_df = pd.read_csv(edges_file, sep='\t')
    if source_col not in edges_df.columns or target_col not in edges_df.columns:
        raise ValueError(f"The edges file must contain columns '{source_col}' and '{target_col}'.")

    # Extract source, target, and all other attributes
    edges = []
    for _, row in edges_df.iterrows():
        edge = {
            "source": row[source_col],
            "target": row[target_col],
            **{k: v for k, v in row.items() if k not in [source_col, target_col]}
        }
        edges.append(edge)

    return edges

def build_graph(nodes, edges):
    """Build a directed graph from nodes and edges."""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    # Add edges with attributes
    for edge in edges:
        source = edge.pop("source")
        target = edge.pop("target")
        G.add_edge(source, target, **edge)

    return G

def main():
    """Main function to parse arguments and build the graph."""
    parser = argparse.ArgumentParser(description="Build a graph from node and edge files.")
    parser.add_argument('--nodes', required=False, help="Path to the nodes TSV file (optional)")
    parser.add_argument('--edges', required=True, help="Path to the edges TSV file")
    parser.add_argument('--output', default='results/graph.graphml', help="Output GraphML file")
    parser.add_argument('--source_col', default='source', help="Column name for the source node in edges file")
    parser.add_argument('--target_col', default='target', help="Column name for the target node in edges file")
    args = parser.parse_args()

    # Load edges
    edges = load_edges(args.edges, source_col=args.source_col, target_col=args.target_col)

    # Load or create nodes
    if args.nodes:
        nodes = load_nodes(args.nodes)
    else:
        # Extract all unique nodes from edges
        all_nodes = set()
        for edge in edges:
            all_nodes.add(edge["source"])
            all_nodes.add(edge["target"])
        nodes = {node: {} for node in all_nodes}

    # Build graph
    G = build_graph(nodes, edges)

    # Save to GraphML
    nx.write_graphml(G, args.output)
    print(f"Graph saved to {args.output}")

if __name__ == '__main__':
    main()
