#!/bin/bash
set -euo pipefail

DIR=figures

# SVG -> PDF (text converted to outlines)
inkscape --export-type=pdf --export-text-to-path "$DIR"/*.svg

# SVG -> PNG at 300 dpi, white background
inkscape --export-type=png --export-dpi=300 \
  --export-background=white --export-background-opacity=1 "$DIR"/*.svg

# PDF -> JPG at 300 dpi, quality 95
for f in "$DIR"/*.svg; do
  pdftoppm -jpeg -jpegopt quality=95 -r 300 -singlefile "${f%.svg}.pdf" "${f%.svg}"
done
