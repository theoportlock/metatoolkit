#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
import math

def parse_dimension(dim_str):
    """
    Parse a dimension string (like '300', '300px', or '216.00pt') to a float.
    Converts points (pt) to pixels using the conversion factor: 1pt ≈ 1.3333px.
    """
    try:
        if dim_str.endswith("px"):
            return float(dim_str[:-2])
        elif dim_str.endswith("pt"):
            return float(dim_str[:-2]) * 1.3333
        else:
            return float(dim_str)
    except Exception as e:
        raise ValueError(f"Cannot parse dimension '{dim_str}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Arrange SVGs into a matrix (grid) by specifying rows or columns."
    )
    parser.add_argument("--rows", type=int, help="Number of rows in the matrix")
    parser.add_argument("--cols", type=int, help="Number of columns in the matrix")
    parser.add_argument("input_svgs", nargs="+", help="Input SVG files")
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file")
    args = parser.parse_args()

    total_svgs = len(args.input_svgs)
    if args.rows is None and args.cols is None:
        parser.error("You must specify either --rows or --cols (or both).")
    
    if args.rows is not None and args.cols is not None:
        # When both are provided, ensure grid is large enough.
        if args.rows * args.cols < total_svgs:
            parser.error(f"Grid of {args.rows} rows and {args.cols} columns ({args.rows * args.cols} cells) is too small for {total_svgs} SVG files.")
        rows = args.rows
        cols = args.cols
    elif args.rows is not None:
        rows = args.rows
        cols = math.ceil(total_svgs / rows)
    else:  # args.cols is not None
        cols = args.cols
        rows = math.ceil(total_svgs / cols)
    
    # Use the first SVG file to determine cell dimensions.
    tree = ET.parse(args.input_svgs[0])
    root = tree.getroot()
    # Attempt to extract width and height attributes (default to 100 if not provided).
    cell_width = parse_dimension(root.get("width", "100"))
    cell_height = parse_dimension(root.get("height", "100"))
    
    # Use viewBox from the first SVG if available, or default to full cell dimensions.
    default_viewBox = root.get("viewBox", f"0 0 {cell_width} {cell_height}")

    # Define overall canvas dimensions.
    main_width = cols * cell_width
    main_height = rows * cell_height

    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    attribs = {
        "width": str(main_width),
        "height": str(main_height),
        "viewBox": f"0 0 {main_width} {main_height}",
        "version": "1.1",
        "xmlns": svg_ns
    }
    main_svg = ET.Element("{" + svg_ns + "}svg", attrib=attribs)

    # Process each SVG file and position it in the grid.
    for idx, filename in enumerate(args.input_svgs):
        row = idx // cols
        col = idx % cols
        x = col * cell_width
        y = row * cell_height

        tree = ET.parse(filename)
        orig_svg = tree.getroot()
        viewBox = orig_svg.get("viewBox", default_viewBox)
        
        nested_attribs = {
            "x": str(x),
            "y": str(y),
            "width": str(cell_width),
            "height": str(cell_height),
            "viewBox": viewBox
        }
        nested_svg = ET.Element("{" + svg_ns + "}svg", attrib=nested_attribs)
        # Append all child elements from the original SVG.
        for child in list(orig_svg):
            nested_svg.append(child)
        main_svg.append(nested_svg)

    ET.ElementTree(main_svg).write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Output written to {args.output}")

if __name__ == "__main__":
    main()
