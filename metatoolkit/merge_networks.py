#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import networkx as nx
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple GraphML files without node overlap.")
    parser.add_argument("graphml_files", nargs='+', help="List of GraphML files to merge")
    parser.add_argument("--output", default="merged.graphml", help="Output merged GraphML filename")
    parser.add_argument("--xgap", type=float, default=10.0, help="Gap between rightmost and leftmost node x-positions")
    parser.add_argument("--directed", action="store_true", help="Use directed graphs")
    return parser.parse_args()

def get_graph_bounds(G):
    """Return the min and max x positions of the graph nodes"""
    positions = nx.get_node_attributes(G, 'x')
    if not positions:
        raise ValueError("Graph does not contain 'x' node attributes")
    x_values = list(positions.values())
    return min(x_values), max(x_values)

def offset_graph(G, x_offset):
    """Offset the x positions of all nodes in the graph by x_offset"""
    for node in G.nodes:
        G.nodes[node]['x'] += x_offset
    return G

def merge_graphs(graph_files, xgap, directed=False):
    GraphType = nx.DiGraph if directed else nx.Graph
    merged = GraphType()
    current_offset = 0.0

    for file in graph_files:
        G = nx.read_graphml(file)

        # Convert graph to the right type
        G = GraphType(G)

        # Ensure x/y positions are float
        for node in G.nodes:
            G.nodes[node]['x'] = float(G.nodes[node].get('x', 0.0))
            G.nodes[node]['y'] = float(G.nodes[node].get('y', 0.0))

        if len(merged.nodes) > 0:
            _, prev_max_x = get_graph_bounds(merged)
            curr_min_x, _ = get_graph_bounds(G)
            offset = (prev_max_x - curr_min_x) + xgap
            G = offset_graph(G, offset)
            current_offset += offset

        # Rename nodes to avoid name collision
        G = nx.relabel_nodes(G, lambda n: f"{os.path.basename(file)}::{n}", copy=True)

        merged = nx.compose(merged, G)

    return merged

def main():
    args = parse_args()
    merged = merge_graphs(args.graphml_files, args.xgap, directed=args.directed)
    nx.write_graphml(merged, args.output)
    print(f"Merged graph saved to {args.output}")

if __name__ == "__main__":
    main()
