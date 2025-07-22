#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP summary plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., summary.pdf)")
    parser.add_argument("--max_display", type=int, default=20, help="Max features to display")
    args = parser.parse_args()

    shap_values = joblib.load(args.input)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, shap_values.data, feature_names=shap_values.feature_names, max_display=args.max_display, show=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

