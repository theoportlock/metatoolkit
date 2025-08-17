#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SHAP force plot as SVG")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output SVG file (e.g., force_sample_0.svg)")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for the force plot")
    return parser.parse_args()

def plot_force_svg(shap_values, sample_index, output_path):
    explanation = shap_values[sample_index]

    # Round for cleaner plot labels
    explanation.values = np.round(explanation.values, 3)
    explanation.base_values = np.round(explanation.base_values, 3)
    explanation.data = np.round(explanation.data, 3)

    # Generate matplotlib-based force plot
    shap.plots.force(explanation, matplotlib=True, show=False)

    # Save as SVG
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    shap_values = joblib.load(args.input)
    plot_force_svg(shap_values, args.sample_index, args.output)

if __name__ == "__main__":
    main()

