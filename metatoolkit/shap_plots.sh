#!/bin/bash

set +e
source env.sh

input="results/prediction/dataset_rf_shap/shap_values_test.joblib"
output="results/prediction/plots"
input=$1
output=$2

SAMPLE_INDEX=0

rm -rf $output
mkdir -p $output

echo "Generating summary plot..."
shap_plot_summary.py \
  --input "$input" \
  --output "$output/summary.svg" \
  --max_display 20

echo "Generating bar plot..."
shap_plot_bar.py \
  --input "$input" \
  --output "$output/bar.svg" \
  --max_display 20

echo "Generating heatmap plot..."
shap_plot_heatmap.py \
  --input "$input" \
  --output "$output/heatmap.svg" \
  --max_display 20

echo "Generating waterfall plot..."
shap_plot_waterfall.py \
  --input "$input" \
  --output "$output/waterfall_sample_${SAMPLE_INDEX}.svg" \
  --sample_index "$SAMPLE_INDEX"

echo "Generating force plot..."
shap_plot_force.py \
  --input "$input" \
  --output "$output/force_sample_${SAMPLE_INDEX}.svg" \
  --sample_index "$SAMPLE_INDEX"

echo "Generating decision plot..."
shap_plot_decision.py \
  --input "$input" \
  --output "$output/decision_sample_${SAMPLE_INDEX}.svg" \
  --sample_index "$SAMPLE_INDEX"

echo "Generating scatter plot..."
shap_plot_scatter.py \
	--input "$input"  \
	--output "$output/scatter.svg" \
	--interaction

echo "All plots saved to $output."

