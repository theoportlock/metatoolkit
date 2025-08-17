#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP waterfall plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., waterfall_sample_0.pdf)")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for the waterfall plot")
    args = parser.parse_args()

    shap_values = joblib.load(args.input)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[args.sample_index], show=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

