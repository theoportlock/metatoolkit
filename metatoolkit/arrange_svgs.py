#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import xml.etree.ElementTree as ET
import math

def parse_dimension(dim_str):
    try:
        if dim_str.endswith("px"):
            return float(dim_str[:-2])
        elif dim_str.endswith("pt"):
            return float(dim_str[:-2]) * 1.3333
        else:
            return float(dim_str)
    except Exception as e:
        raise ValueError(f"Cannot parse dimension '{dim_str}': {e}")

def clean_svg_root(svg_root):
    """Remove problematic attributes (xmlns, version, etc.) from nested SVGs."""
    for attr in ["xmlns", "xmlns:xlink", "version"]:
        if attr in svg_root.attrib:
            del svg_root.attrib[attr]
    return svg_root

def main():
    parser = argparse.ArgumentParser(description="Arrange SVGs into a grid while preserving aspect ratio and font sizes.")
    parser.add_argument("--rows", type=int, help="Number of rows in the grid")
    parser.add_argument("--cols", type=int, help="Number of columns in the grid")
    parser.add_argument("input_svgs", nargs="+", help="Input SVG files")
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file")
    args = parser.parse_args()

    total_svgs = len(args.input_svgs)
    if args.rows is None and args.cols is None:
        parser.error("You must specify either --rows or --cols (or both).")

    if args.rows and args.cols:
        if args.rows * args.cols < total_svgs:
            parser.error(f"Grid of {args.rows}x{args.cols} too small for {total_svgs} SVGs.")
        rows, cols = args.rows, args.cols
    elif args.rows:
        rows = args.rows
        cols = math.ceil(total_svgs / rows)
    else:
        cols = args.cols
        rows = math.ceil(total_svgs / cols)

    # Parse all SVGs and store their dimensions
    svgs = []
    max_width = 0
    max_height = 0
    for filename in args.input_svgs:
        tree = ET.parse(filename)
        root = tree.getroot()
        viewBox = root.get("viewBox")
        if viewBox:
            _, _, w, h = map(float, viewBox.strip().split())
            width, height = w, h
        else:
            width = parse_dimension(root.get("width", "100"))
            height = parse_dimension(root.get("height", "100"))
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        svgs.append((tree, width, height))

    canvas_width = cols * max_width
    canvas_height = rows * max_height

    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    root_svg = ET.Element(
        "{" + svg_ns + "}svg",
        attrib={
            "width": str(canvas_width),
            "height": str(canvas_height),
            "viewBox": f"0 0 {canvas_width} {canvas_height}",
            "version": "1.1",
            "xmlns": svg_ns
        }
    )

    for idx, (tree, width, height) in enumerate(svgs):
        row = idx // cols
        col = idx % cols
        x_offset = col * max_width
        y_offset = row * max_height

        orig_root = clean_svg_root(tree.getroot())

        # Wrap contents in a group with translation transform
        group = ET.Element("{" + svg_ns + "}g", attrib={"transform": f"translate({x_offset},{y_offset})"})

        # Copy children safely
        for child in list(orig_root):
            group.append(child)

        root_svg.append(group)

    ET.ElementTree(root_svg).write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Output written to {args.output}")

if __name__ == "__main__":
    main()

