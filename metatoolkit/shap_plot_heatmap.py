#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP heatmap plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., heatmap.pdf)")
    parser.add_argument("--max_display", type=int, default=20, help="Max features to display")
    args = parser.parse_args()

    shap_values = joblib.load(args.input)
    plt.figure(figsize=(10, 8))
    shap.plots.heatmap(shap_values[:,:args.max_display], show=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

