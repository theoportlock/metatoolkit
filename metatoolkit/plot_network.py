#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import textwrap
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate network graph PDF/SVG from GraphML')
    parser.add_argument('graphml_file', help='Input GraphML file')
    parser.add_argument('--output_file', default='network.svg', help='Output PDF or SVG file')
    parser.add_argument('--edge_color_attr', default=None, help='Edge attribute to use for coloring edges')
    parser.add_argument('--node_color_attr', default=None, help='Node attribute to use for coloring nodes')
    parser.add_argument('--cmap', default='coolwarm', help='Matplotlib colormap to use (default: coolwarm)')
    parser.add_argument('--layout', default='spring', choices=['spring', 'kamada_kawai', 'planar', 'circular', 'random', 'shell'], help='Layout algorithm to use')
    parser.add_argument('--node_size', type=int, default=300, help='Size of nodes (default: 300)')
    parser.add_argument('--font_size', type=int, default=5, help='Font size for labels (default: 5)')
    parser.add_argument('--max_label_width', type=int, default=150, help='Maximum label width in pixels (default: 150)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8,8], help='Width and height of the figure in inches (default: 8 8)')
    return parser.parse_args()

def plot_network(graphml_file, output_file, edge_color_attr=None, node_color_attr=None, cmap_name='coolwarm', layout='spring', node_size=300, font_size=5, max_label_width=150, figsize=(8,8)):
    G = nx.read_graphml(graphml_file)
        
    # Determine node positions
    if all('x' in G.nodes[node] and 'y' in G.nodes[node] for node in G.nodes):
        pos = {node: (float(G.nodes[node]['x']), -float(G.nodes[node]['y'])) for node in G.nodes}
    else:
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'planar':
            pos = nx.planar_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            raise ValueError(f"Unsupported layout: {layout}")
    
    # Create wrapped labels
    def wrap_label(text, max_pixels, font_size=6):
        max_chars = int(max_pixels / (font_size * 0.6))
        return "\n".join(textwrap.wrap(text, max_chars))

    labels = {node: wrap_label(G.nodes[node].get('label', node), max_label_width, font_size=font_size) for node in G.nodes}

    # Handle node colors
    node_colors = 'lightgray'
    if node_color_attr:
        node_values = []
        for node, d in G.nodes(data=True):
            val = d.get(node_color_attr, None)
            node_values.append(float(val) if val is not None else 0.0)
        
        norm = colors.Normalize(vmin=min(node_values), vmax=max(node_values))
        cmap = cm.get_cmap(cmap_name)
        node_colors = [cmap(norm(val)) for val in node_values]

    # Handle edge colors
    edge_colors = 'lightgray'
    if edge_color_attr:
        edge_values = []
        for u, v, d in G.edges(data=True):
            val = d.get(edge_color_attr, None)
            edge_values.append(float(val) if val is not None else 0.0)
        
        norm = colors.Normalize(vmin=min(edge_values), vmax=max(edge_values))
        cmap = cm.get_cmap(cmap_name)
        edge_colors = [cmap(norm(val)) for val in edge_values]
    
    # Draw and save graph
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        labels=labels,
        with_labels=True,
        node_size=node_size,
        font_size=font_size,
        arrows=False,
        node_color=node_colors,
        edge_color=edge_colors
    )
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

def main():
    args = parse_arguments()
    plot_network(
        graphml_file=args.graphml_file,
        output_file=args.output_file,
        edge_color_attr=args.edge_color_attr,
        node_color_attr=args.node_color_attr,
        cmap_name=args.cmap,
        layout=args.layout,
        node_size=args.node_size,
        font_size=args.font_size,
        max_label_width=args.max_label_width,
        figsize=tuple(args.figsize)
    )

if __name__ == "__main__":
    main()
