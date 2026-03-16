#!/usr/bin/env bash
set -euo pipefail

# ========================
# Generic MOFA CLI pipeline
# ========================

# Usage:
#   mofa_tool.sh \
#       --data-dir results/filtered \
#       --output-dir results/mofa_metabolome \
#       --datasets lipids.tsv vitamin.tsv aa.tsv \
#       --merge vitamin.tsv aa.tsv \
#       --meta-file results/filtered/meta.tsv \
#       --time-meta results/filtered/timemeta_0_52tp.tsv \
#       --time-col timepoint \
#       --reindex subjectID \
#       --factors 20
#
# Optional flags:
#   --no-merge     Skip merging step

# ----------------
# Parse arguments
# ----------------
DATA_DIR=""
OUTPUT_DIR=""
META_FILE=""
TIME_META=""
TIME_COL="timepoint"
REINDEX="subjectID"
FACTORS=20
MERGE_DATASETS=()
DATASETS=()
DO_MERGE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --meta-file) META_FILE="$2"; shift 2 ;;
    --time-meta) TIME_META="$2"; shift 2 ;;
    --time-col) TIME_COL="$2"; shift 2 ;;
    --reindex) REINDEX="$2"; shift 2 ;;
    --factors) FACTORS="$2"; shift 2 ;;
    --datasets)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --merge)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MERGE_DATASETS+=("$1")
        shift
      done
      ;;
    --no-merge)
      DO_MERGE=false
      shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

# ---------------------
# Check required inputs
# ---------------------
if [[ -z "$DATA_DIR" || -z "$OUTPUT_DIR" || -z "$META_FILE" || -z "$TIME_META" ]]; then
  echo "Missing required arguments. Use --data-dir, --output-dir, --meta-file, and --time-meta."
  exit 1
fi

mkdir -p "$OUTPUT_DIR/data" "$OUTPUT_DIR/timepoints"

# ----------------
# Step 1: Scale data
# ----------------
echo "🔹 Scaling datasets..."
for dataset in "${DATASETS[@]}"; do
  name=$(basename "$dataset" .tsv)
  echo "  - $dataset -> ${name}_CLR.tsv"
  scale.py CLR "$DATA_DIR/$dataset" --output "$OUTPUT_DIR/data/${name}_CLR.tsv"
done

# ----------------
# Step 2: Merge datasets
# ----------------
if $DO_MERGE && [[ ${#MERGE_DATASETS[@]} -gt 0 ]]; then
  echo "🔹 Merging datasets: ${MERGE_DATASETS[*]}"
  merged_name="merged"
  merge_inputs=()
  for m in "${MERGE_DATASETS[@]}"; do
    name=$(basename "$m" .tsv)
    merge_inputs+=("$OUTPUT_DIR/data/${name}_CLR.tsv")
  done
  merge.py "${merge_inputs[@]}" -o "$OUTPUT_DIR/data/${merged_name}_CLR.tsv"
  DATASETS+=("${merged_name}.tsv")
fi

# ----------------
# Step 3: Split by timepoints
# ----------------
echo "🔹 Splitting by timepoints..."
for dataset in "${DATASETS[@]}"; do
  name=$(basename "$dataset" .tsv)
  splitter.py \
    "$OUTPUT_DIR/data/${name}_CLR.tsv" \
    -m "$TIME_META" \
    -col "$TIME_COL" \
    --outdir "$OUTPUT_DIR/timepoints" \
    --reindex "$REINDEX"
done

# ----------------
# Step 4: Load into MOFA
# ----------------
echo "🔹 Loading data into MOFA..."
mofa_load_data.py \
  --t1-dir "$OUTPUT_DIR/timepoints/0" \
  --meta-file "$META_FILE" \
  --output "$OUTPUT_DIR/mdata.h5mu"

# ----------------
# Step 5: Run MOFA
# ----------------
echo "🔹 Running MOFA with $FACTORS factors..."
mofa_run.py \
  --input "$OUTPUT_DIR/mdata.h5mu" \
  --factors "$FACTORS" \
  --output "$OUTPUT_DIR/mofa_model.hdf5"

# ----------------
# Step 6: Export results
# ----------------
echo "🔹 Exporting MOFA results..."
mofa_export.R \
  --model "$OUTPUT_DIR/mofa_model.hdf5" \
  --outdir "$OUTPUT_DIR/extract"

cp "$OUTPUT_DIR/extract/sample_factors.tsv" \
   "$OUTPUT_DIR/mofa_results.tsv"

echo "✅ MOFA pipeline completed successfully!"
echo "Results saved to: $OUTPUT_DIR/mofa_results.tsv"

