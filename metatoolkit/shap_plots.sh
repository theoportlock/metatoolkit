#!/bin/bash

# --- Configuration ---
SHAP_VALUES_INPUT="results/dataset_rf_shap/shap_values_test.joblib"
SHAP_INTERACTION_VALUES_INPUT="results/dataset_rf_shap/shap_interaction_values_test.joblib"
OUTPUT_DIR="results/shap_plots_test_data"

SAMPLE_INDEX=0

mkdir -p "$OUTPUT_DIR"

echo "Generating summary plot..."
python3 code/shap_plot_summary.py \
  --input "$SHAP_VALUES_INPUT" \
  --output "$OUTPUT_DIR/summary.svg" \
  --max_display 20

echo "Generating bar plot..."
python3 code/shap_plot_bar.py \
  --input "$SHAP_VALUES_INPUT" \
  --output "$OUTPUT_DIR/bar.svg" \
  --max_display 20

echo "Generating heatmap plot..."
python3 code/shap_plot_heatmap.py \
  --input "$SHAP_VALUES_INPUT" \
  --output "$OUTPUT_DIR/heatmap.svg" \
  --max_display 20

echo "Generating waterfall plot..."
python3 code/shap_plot_waterfall.py \
  --input "$SHAP_VALUES_INPUT" \
  --output "$OUTPUT_DIR/waterfall_sample_${SAMPLE_INDEX}.svg" \
  --sample_index "$SAMPLE_INDEX"

#echo "Generating force plot..."
#python3 code/shap_plot_force.py \
#  --input "$SHAP_VALUES_INPUT" \
#  --output "$OUTPUT_DIR/force_sample_${SAMPLE_INDEX}.svg" \
#  --sample_index "$SAMPLE_INDEX"

#echo "Generating decision plot..."
#python3 code/shap_plot_decision.py \
#  --input "$SHAP_VALUES_INPUT" \
#  --output "$OUTPUT_DIR/decision_sample_${SAMPLE_INDEX}.svg" \
#  --sample_index "$SAMPLE_INDEX"

echo "Generating scatter plot..."
python3 code/shap_plot_scatter.py \
	--input "$SHAP_VALUES_INPUT"  \
	--output "$OUTPUT_DIR/scatter.svg" \
	--interaction

echo "All plots saved to $OUTPUT_DIR."

