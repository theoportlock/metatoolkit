#!/usr/bin/env python

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

def plot_network(graphml_file, output_file, max_label_width=150):
    # Read the graph from the GraphML file
    G = nx.read_graphml(graphml_file)

    # Extract node positions if they exist in the GraphML file, flipping the y-coordinate
    if all('x' in G.nodes[node] and 'y' in G.nodes[node] for node in G.nodes):
        pos = {node: (float(G.nodes[node]['x']), -float(G.nodes[node]['y'])) for node in G.nodes}
    else:
        pos = nx.spring_layout(G)  # Default layout if no positions are specified in the file

    # Check if nodes have 'status' attribute; if so, use it for node color
    if all('status' in G.nodes[node] for node in G.nodes):
        node_color = [G.nodes[node]['status'] for node in G.nodes]
    else:
        node_color = 'blue'  # Default color if no status info is available

    # Function to wrap text based on max pixel width (approximate)
    def wrap_label(text, max_pixels, font_size=6):
        # Approximate width in characters for given pixel width
        max_chars = int(max_pixels / (font_size * 0.6))  # Estimate based on font size
        wrapped_text = "\n".join(textwrap.wrap(text, max_chars))
        return wrapped_text

    # Use the 'label' attribute for node labels if it exists, wrapped to max width
    labels = {node: wrap_label(G.nodes[node].get('label', node), max_label_width) for node in G.nodes}

    # Plot the network with a smaller font size and save to PDF
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, labels=labels, with_labels=True,
            node_color=node_color, cmap=plt.cm.viridis,
            font_size=6, font_color='black')  # Smaller font size for labels
    plt.savefig(output_file, format='pdf')
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot a network graph from a GraphML file and save it as a PDF.')
    parser.add_argument('--graphml-file', required=True, help='Path to the GraphML file')
    parser.add_argument('--output-file', required=True, help='Path to save the output PDF file')
    args = parser.parse_args()

    # Plot the network and save as PDF
    plot_network(args.graphml_file, args.output_file)

