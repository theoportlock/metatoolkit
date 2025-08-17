#!/usr/bin/env python3
import argparse
import joblib
import shap
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SHAP dependence scatter or interaction plot")
    parser.add_argument("--input", required=True, help="Path to SHAP Explanation joblib file")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., scatter_plot.pdf)")
    parser.add_argument("--feature", help="Feature name to plot on the x-axis")
    parser.add_argument("--interaction", action="store_true", help="Enable interaction coloring using strongest feature")
    return parser.parse_args()


def plot_shap_scatter(shap_values, output_path, feature=None, interaction=False):
    # Use feature if specified, otherwise default to first feature
    shap_feat = shap_values[:, feature] if feature else shap_values[:, 0]

    # Use interaction coloring (auto-select strongest interacting feature)
    color = shap_values if interaction else shap_feat

    shap.plots.scatter(shap_feat, color=color, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved SHAP scatter plot to {output_path}")


def main():
    args = parse_args()
    shap_values = joblib.load(args.input)

    if shap_values.values.ndim == 3 and not args.interaction:
        raise ValueError(
            "Loaded SHAP values appear to be interaction values (3D), "
            "but --interaction flag was not set. Use --interaction to plot them."
        )

    plot_shap_scatter(
        shap_values,
        output_path=args.output,
        feature=args.feature,
        interaction=args.interaction
    )


if __name__ == "__main__":
    main()

