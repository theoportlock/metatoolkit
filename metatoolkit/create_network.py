#!/usr/bin/env python
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

def load_edges(edges_file):
    """Load edges from a TSV file, assuming the first two columns are source and target."""
    edges_df = pd.read_csv(edges_file, sep='\t')
    if edges_df.shape[1] < 2:
        raise ValueError("The edges file must have at least two columns for source and target.")

    # Extract source, target, and all other attributes
    source_target = edges_df.iloc[:, :2].values  # First two columns as source and target
    attributes = edges_df.iloc[:, 2:]  # Remaining columns as attributes

    edges = []
    for (source, target), attr_row in zip(source_target, attributes.to_dict(orient='records')):
        edge = {"source": source, "target": target, **attr_row}
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
    parser.add_argument('--nodes', required=True, help="Path to the nodes TSV file")
    parser.add_argument('--edges', required=True, help="Path to the edges TSV file")
    parser.add_argument('--output', default='graph.graphml', help="Output GraphML file")
    args = parser.parse_args()

    # Load nodes and edges
    nodes = load_nodes(args.nodes)
    edges = load_edges(args.edges)

    # Build graph
    G = build_graph(nodes, edges)

    # Save to GraphML
    nx.write_graphml(G, args.output)
    print(f"Graph saved to {args.output}")

if __name__ == '__main__':
    main()
