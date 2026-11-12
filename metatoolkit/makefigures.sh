#!/usr/bin/env bash
#
# export-figures.sh - Convert SVG figures to PDF (text outlined), PNG and JPG.
#
# Requires: inkscape (1.x), and pdftoppm (poppler-utils) for JPG output.

set -euo pipefail

PROG=$(basename "$0")

# ---------------------------------------------------------------- defaults --
OUTDIR=""
FORMATS="pdf,png,jpg"
DPI=300
QUALITY=95
BACKGROUND="white"
INPUTS=()

# ----------------------------------------------------------------- helpers --
usage() {
  cat <<EOF
Usage: $PROG [options] [INPUT...]

Convert SVG figures to PDF (text converted to outlines), PNG and JPG.

INPUT is one or more SVG files or directories containing SVGs
(default: ./figures). Directories are not searched recursively.

Options:
  -i, --input PATH      SVG file or directory (repeatable; same as a positional INPUT)
  -o, --output DIR      Output directory (default: next to each input SVG)
  -f, --formats LIST    Comma-separated list of: pdf,png,jpg (default: $FORMATS)
  -d, --dpi N           Resolution for PNG and JPG (default: $DPI)
  -q, --quality N       JPG quality, 1-100 (default: $QUALITY)
  -b, --background COL  PNG background colour (default: $BACKGROUND)
  -h, --help            Show this help and exit

Examples:
  $PROG                                    # figures/*.svg, outputs beside the SVGs
  $PROG -i figures -o build/figures        # write everything to build/figures
  $PROG -f pdf,png -o out fig1.svg fig2.svg
  $PROG -d 600 -q 90 figures
EOF
}

die() {
  echo "$PROG: $*" >&2
  exit 1
}

need_arg() {
  # need_arg OPTION ARG_COUNT_REMAINING
  [[ $2 -ge 2 ]] || die "option $1 requires an argument"
}

# ------------------------------------------------------------ parse options --
while (( $# )); do
  case $1 in
    -i|--input)       need_arg "$1" "$#"; INPUTS+=("$2");  shift 2 ;;
    -o|--output)      need_arg "$1" "$#"; OUTDIR=$2;       shift 2 ;;
    -f|--formats)     need_arg "$1" "$#"; FORMATS=$2;      shift 2 ;;
    -d|--dpi)         need_arg "$1" "$#"; DPI=$2;          shift 2 ;;
    -q|--quality)     need_arg "$1" "$#"; QUALITY=$2;      shift 2 ;;
    -b|--background)  need_arg "$1" "$#"; BACKGROUND=$2;   shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; INPUTS+=("$@"); break ;;
    -*)               die "unknown option: $1 (see --help)" ;;
    *)                INPUTS+=("$1"); shift ;;
  esac
done

(( ${#INPUTS[@]} )) || INPUTS=(figures)

# ----------------------------------------------------------------- validate --
[[ $DPI =~ ^[0-9]+$ && $DPI -gt 0 ]]           || die "--dpi must be a positive integer"
[[ $QUALITY =~ ^[0-9]+$ && $QUALITY -ge 1 && $QUALITY -le 100 ]] \
                                               || die "--quality must be between 1 and 100"

WANT_PDF=0; WANT_PNG=0; WANT_JPG=0
IFS=',' read -r -a fmt_list <<< "$FORMATS"
for fmt in "${fmt_list[@]}"; do
  case ${fmt,,} in
    pdf) WANT_PDF=1 ;;
    png) WANT_PNG=1 ;;
    jpg|jpeg) WANT_JPG=1 ;;
    *) die "unknown format '$fmt' (choose from: pdf, png, jpg)" ;;
  esac
done
(( WANT_PDF || WANT_PNG || WANT_JPG )) || die "no output formats selected"

command -v inkscape >/dev/null 2>&1 || die "inkscape not found in PATH"
if (( WANT_JPG )); then
  command -v pdftoppm >/dev/null 2>&1 || die "pdftoppm not found (install poppler-utils)"
fi

# -------------------------------------------------------- collect SVG files --
SVGS=()
for p in "${INPUTS[@]}"; do
  if [[ -d $p ]]; then
    while IFS= read -r -d '' f; do
      SVGS+=("$f")
    done < <(find "$p" -maxdepth 1 -type f -name '*.svg' -print0 | sort -z)
  elif [[ -f $p ]]; then
    [[ $p == *.svg ]] || die "not an SVG file: $p"
    SVGS+=("$p")
  else
    die "input not found: $p"
  fi
done
(( ${#SVGS[@]} )) || die "no SVG files found in: ${INPUTS[*]}"

[[ -z $OUTDIR ]] || mkdir -p "$OUTDIR"

# Scratch space for intermediate PDFs when JPG is requested without PDF
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# ------------------------------------------------------------------ convert --
for svg in "${SVGS[@]}"; do
  base=$(basename "${svg%.svg}")
  out_dir=${OUTDIR:-$(dirname "$svg")}
  stem="$out_dir/$base"

  echo "==> $svg"

  # SVG -> PDF (text converted to outlines). Also the source for JPG.
  pdf=""
  if (( WANT_PDF )); then
    pdf="$stem.pdf"
  elif (( WANT_JPG )); then
    pdf="$WORKDIR/$base.pdf"
  fi
  if [[ -n $pdf ]]; then
    inkscape --export-type=pdf --export-text-to-path \
      --export-filename="$pdf" "$svg"
  fi

  # SVG -> PNG at the requested dpi on an opaque background
  if (( WANT_PNG )); then
    inkscape --export-type=png --export-dpi="$DPI" \
      --export-background="$BACKGROUND" --export-background-opacity=1 \
      --export-filename="$stem.png" "$svg"
  fi

  # PDF -> JPG at the requested dpi and quality
  if (( WANT_JPG )); then
    pdftoppm -jpeg -jpegopt "quality=$QUALITY" -r "$DPI" -singlefile \
      "$pdf" "$stem"
  fi
done

echo "Done: ${#SVGS[@]} figure(s) processed."
