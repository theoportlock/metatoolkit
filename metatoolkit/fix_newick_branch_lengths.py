#!/usr/bin/env python3

import argparse
import logging
from skbio import TreeNode


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ensure all non-root nodes in a Newick tree have branch lengths. "
            "Missing lengths are filled with a default value."
        )
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input Newick file"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output Newick file"
    )

    parser.add_argument(
        "--default-length",
        type=float,
        default=1.0,
        help="Default branch length to assign when missing (default: 1.0)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def fix_branch_lengths(tree: TreeNode, default_length: float) -> int:
    """
    Assign default branch lengths to nodes missing them.

    Returns:
        int: number of nodes modified
    """
    missing_nodes = [
        n for n in tree.postorder()
        if not n.is_root() and n.length is None
    ]

    for node in missing_nodes:
        node.length = default_length

    return len(missing_nodes)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logging.info(f"Loading tree: {args.input}")
    tree = TreeNode.read(args.input)

    logging.info("Checking for missing branch lengths...")
    n_fixed = fix_branch_lengths(tree, args.default_length)

    if n_fixed == 0:
        logging.info("No missing branch lengths found.")
    else:
        logging.info(f"Fixed {n_fixed} nodes with missing branch lengths.")

    logging.info(f"Writing fixed tree to: {args.output}")
    tree.write(args.output)

    logging.info("Done.")


if __name__ == "__main__":
    main()
