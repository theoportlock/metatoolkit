#!/usr/bin/env bash
# =====================================================
# A generic pipeline for evaluating model power and SHAP
# Theo Portlock, 2025
# =====================================================

set -euo pipefail
source env.sh

# ------------------------
# Usage example
# ------------------------
# model_power_tool.sh \
#   --input results/merged_dataset.tsv \
#   --output results/prediction \
#   --target WLZ_WHZ \
#   --y-file results/timepoints/yr2/anthro.tsv \
#   --model random_forest \
#   --task regression \
#   --layout shell \
#   --cmap Reds \
#   --figsize 4 4 \
#   --cols 2
#
# Supported models: random_forest, xgboost
# ------------------------

# ------------------------
# Default values
# ------------------------
INPUT=""
OUTPUT=""
Y_COL="WLZ_WHZ"
Y_FILE=""
MODEL="random_forest"
TASK="regression"
SCALER="none"
KEEPNA=true
LAYOUT="shell"
CMAP="Reds"
FIGSIZE_X=4
FIGSIZE_Y=4
COLS=2

# ------------------------
# Parse arguments
# ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --target|--y-col) Y_COL="$2"; shift 2 ;;
    --y-file) Y_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --scaler) SCALER="$2"; shift 2 ;;
    --no-keepna) KEEPNA=false; shift ;;
    --layout) LAYOUT="$2"; shift 2 ;;
    --cmap) CMAP="$2"; shift 2 ;;
    --figsize) FIGSIZE_X="$2"; FIGSIZE_Y="$3"; shift 3 ;;
    --cols) COLS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

# ------------------------
# Validate required args
# ------------------------
if [[ -z "$INPUT" || -z "$OUTPUT" || -z "$Y_FILE" ]]; then
  echo "❌ Missing required arguments: --input, --output, and --y-file are mandatory."
  exit 1
fi

# ------------------------
# Setup directories
# ------------------------
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

echo "🔹 Starting model power pipeline"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Target column: $Y_COL"
echo "Y-file: $Y_FILE"
echo

# ------------------------
# Step 1: Train/test split
# ------------------------
echo "🔹 Splitting dataset..."
KEEPNA_FLAG=""
$KEEPNA && KEEPNA_FLAG="--keepna"

test_train_split.py \
  --input "$INPUT" \
  --y_col "$Y_COL" \
  --y_file "$Y_FILE" \
  --scaler "$SCALER" \
  $KEEPNA_FLAG \
  --output_dir "$OUTPUT/dataset_split"

# ------------------------
# Step 2: Train model
# ------------------------
echo "🔹 Training model ($MODEL)..."
if [[ "$MODEL" == "random_forest" ]]; then
  random_forest.py \
    --input_dir "$OUTPUT/dataset_split/" \
    --task "$TASK" \
    --output_model "$OUTPUT/model.pkl"
elif [[ "$MODEL" == "xgboost" ]]; then
  xgboost_model.py \
    --input_dir "$OUTPUT/dataset_split/" \
    --task "$TASK" \
    --output_model "$OUTPUT/model.pkl"
else
  echo "❌ Unsupported model type: $MODEL"
  exit 1
fi

# ------------------------
# Step 3: Evaluate model
# ------------------------
echo "🔹 Evaluating model..."
evaluate_model.py \
  --model "$OUTPUT/model.pkl" \
  --input_dir "$OUTPUT/dataset_split/" \
  --task "$TASK" \
  --report_file "$OUTPUT/model_report.tsv"

# ------------------------
# Step 4: Compute SHAP values
# ------------------------
echo "🔹 Calculating SHAP values..."
shap_interpret.py \
  --model "$OUTPUT/model.pkl" \
  --input_dir "$OUTPUT/dataset_split/" \
  --shap_val \
  --shap_interact \
  --output_dir "$OUTPUT/model_shap"

# ------------------------
# Step 5: Create interaction network
# ------------------------
echo "🔹 Building SHAP interaction network..."
create_network.py \
  --edges "$OUTPUT/model_shap/mean_abs_shap_interaction_train.tsv" \
  --output "$OUTPUT/network.graphml"

plot_network.py \
  "$OUTPUT/network.graphml" \
  --edge_color_attr mean_abs_shap_interaction_test.tsv \
  --layout "$LAYOUT" \
  --cmap "$CMAP" \
  --figsize "$FIGSIZE_X" "$FIGSIZE_Y" \
  --output "$OUTPUT/network.svg"

# ------------------------
# Step 6: SHAP plots
# ------------------------
echo "🔹 Generating SHAP plots..."
shap_plots.sh \
  "$OUTPUT/model_shap/shap_values_test.joblib" \
  "$OUTPUT/plots"

# ------------------------
# Step 7: Residuals plot
# ------------------------
echo "🔹 Plotting regression residuals..."
plot_regression_residuals.py \
  --model "$OUTPUT/model.pkl" \
  --input "$OUTPUT/dataset_split" \
  --output "$OUTPUT/plots"

# ------------------------
# Step 8: Merge SHAP plots
# ------------------------
echo "🔹 Arranging SHAP plots into grid..."
arrange_svgs.py \
  "$OUTPUT/plots/"* \
  --cols "$COLS" \
  --output "$OUTPUT/shap_plots_merged.svg"

# ------------------------
# Done
# ------------------------
echo "✅ Pipeline complete!"
echo "Results saved in: $OUTPUT"
echo
ls -1 "$OUTPUT"

