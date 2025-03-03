#!/usr/bin/env python

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

import metatoolkit.functions as f

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate acyclic network graph PDF from GraphML')
    parser.add_argument('graphml_file', help='Input GraphML file')
    parser.add_argument('--output_file', default='network.pdf', help='Output PDF file')
    return parser.parse_args()

def plot_network(graphml_file, output_file, max_label_width=150):
    G = nx.read_graphml(graphml_file)
        
    # Get node positions or generate layout
    if all('x' in G.nodes[node] and 'y' in G.nodes[node] for node in G.nodes):
        pos = {node: (float(G.nodes[node]['x']), -float(G.nodes[node]['y'])) for node in G.nodes}
    else:
        pos = nx.spring_layout(G)
    
    # Create wrapped labels
    def wrap_label(text, max_pixels, font_size=6):
        max_chars = int(max_pixels / (font_size * 0.6))
        return "\n".join(textwrap.wrap(text, max_chars))

    labels = {node: wrap_label(G.nodes[node].get('label', node), max_label_width) for node in G.nodes}
    
    # Draw and save graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=300, font_size=5, arrows=False, node_color='lightgray', edge_color='lightgray')
    plt.tight_layout()
    f.savefig(output_file)
  
def main():
    args = parse_arguments()
    plot_network(args.graphml_file, args.output_file)

if __name__ == "__main__":
    main()

