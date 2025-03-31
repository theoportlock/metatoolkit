#!/usr/bin/env python3
import argparse
import networkx as nx

def main():
    parser = argparse.ArgumentParser(
        description="Compute the shortest path between two nodes in a GraphML file and save it to a file."
    )
    parser.add_argument("graphml", help="Path to the GraphML file.")
    parser.add_argument("source", help="Identifier of the source node.")
    parser.add_argument("target", help="Identifier of the target node.")
    parser.add_argument("output", help="Path to the output file.")
    parser.add_argument(
        "--weight",
        help="Edge attribute to use as weight (if applicable). Omit for unweighted shortest path.",
        default=None,
    )
    args = parser.parse_args()

    # Load the graph from the GraphML file.
    try:
        G = nx.read_graphml(args.graphml)
    except Exception as e:
        print(f"Error reading GraphML file: {e}")
        return

    # Compute the shortest path.
    try:
        path = nx.shortest_path(G, source=args.source, target=args.target, weight=args.weight)
        with open(args.output, "w") as f:
            f.write("\n".join(path) + "\n")
        print(f"Shortest path saved to {args.output}")
    except nx.NetworkXNoPath:
        print(f"No path found between {args.source} and {args.target}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
