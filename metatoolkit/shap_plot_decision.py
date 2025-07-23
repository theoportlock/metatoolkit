#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP decision plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., decision_sample_0.pdf)")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for decision plot")
    args = parser.parse_args()

    shap_values = joblib.load(args.input)

    expected_value = shap_values.base_values[args.sample_index] if hasattr(shap_values, 'base_values') else 0

    plt.figure(figsize=(10, 6))
    shap.decision_plot(expected_value, shap_values.values[args.sample_index], feature_names=shap_values.feature_names, show=False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

