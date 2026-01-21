#!/usr/bin/env bash
set -euo pipefail

source env.sh

usage() {
    cat <<EOF
Usage:
  $(basename "$0") <input.tsv> <meta.tsv> <output_dir> [options]

Positional arguments:
  input.tsv        Beta diversity (or similar) table
  meta.tsv         Metadata table
  output_dir       Output directory

Options:
  -s, --sample-col COL     Sample ID column in metadata (default: sampleID)
  -j, --subject-col COL    Subject ID column in metadata (default: subjectID)
  -t, --timepoint-col COL  Timepoint column in metadata (default: timepoint)
  -b, --baseline TP        Baseline timepoint value (default: PCV2)
  -h, --help               Show this help message
EOF
}

# Defaults
SAMPLE_COL="sampleID"
SUBJECT_COL="subjectID"
TIMEPOINT_COL="timepoint"
BASELINE_TP="PCV2"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sample-col)
            SAMPLE_COL="$2"
            shift 2
            ;;
        -j|--subject-col)
            SUBJECT_COL="$2"
            shift 2
            ;;
        -t|--timepoint-col)
            TIMEPOINT_COL="$2"
            shift 2
            ;;
        -b|--baseline)
            BASELINE_TP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Positional args
if [[ $# -ne 3 ]]; then
    usage
    exit 1
fi

INPUT="$1"
META="$2"
OUTPUT="$3"

mkdir -p "$OUTPUT"

# Join source → metadata
join.py "$INPUT" \
    "$META" \
    --left_on source \
    --right_on "$SAMPLE_COL" \
    -o "$OUTPUT/beta_source.tsv"

# Join target → metadata
join.py "$OUTPUT/beta_source.tsv" \
    "$META" \
    --left_on target \
    --right_on "$SAMPLE_COL" \
    -o "$OUTPUT/beta_source_target.tsv"

# Filter to baseline timepoint
filter.py \
    "$OUTPUT/beta_source_target.tsv" \
    -q "${TIMEPOINT_COL}_x == \"$BASELINE_TP\"" \
    -o "$OUTPUT/beta_source_target_baseline.tsv"

# Keep same-subject pairs
filter.py \
    "$OUTPUT/beta_source_target_baseline.tsv" \
    -q "${SUBJECT_COL}_x == ${SUBJECT_COL}_y" \
    -o "$OUTPUT/beta_source_target_baseline_samesubj.tsv"

