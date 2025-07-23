#!/usr/bin/env python
import argparse
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os


def load_shap_data(input_path):
    data = joblib.load(input_path)

    if isinstance(data, shap.Explanation):
        return data

    raise ValueError("Expected SHAP Explanation object. Got: {}".format(type(data)))


def plot_summary(shap_data, output_path, max_display):
    shap.plots.beeswarm(shap_data, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved summary plot to {output_path}")


def plot_bar(shap_data, output_path, max_display):
    shap.plots.bar(shap_data, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved bar plot to {output_path}")


def plot_heatmap(shap_data, output_path, max_display):
    shap.plots.heatmap(shap_data, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved heatmap to {output_path}")


def plot_waterfall(shap_data, sample_index, output_path):
    shap.plots.waterfall(shap_data[sample_index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved waterfall plot for sample {sample_index} to {output_path}")


def plot_force(shap_data, sample_index, output_path):
    shap.plots.force(shap_data[sample_index], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved force plot for sample {sample_index} to {output_path}")


def plot_decision(shap_data, sample_index, output_path):
    if not isinstance(shap_data, shap.Explanation):
        raise ValueError("Expected shap_data to be a shap.Explanation object")

    expected_value = shap_data.base_values
    if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
        expected_value = expected_value[1]

    features = shap_data.data
    shap_values = shap_data.values

    shap.decision_plot(
        base_value=expected_value,
        shap_values=shap_values,
        features=features,
        feature_names=shap_data.feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved decision plot to {output_path}")


def plot_interaction_scatter(shap_data, output_path):
    shap.plots.scatter(shap_data, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved interaction scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input SHAP values .joblib")
    parser.add_argument("--output", required=True, help="Path to save the plot")
    parser.add_argument("--plot_type", required=True, help="Type of SHAP plot")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index for individual plots")
    parser.add_argument("--max_display", type=int, default=10, help="Maximum features to display")

    args = parser.parse_args()

    shap_data = load_shap_data(args.input)

    if args.plot_type == "summary":
        plot_summary(shap_data, args.output, args.max_display)
    elif args.plot_type == "bar":
        plot_bar(shap_data, args.output, args.max_display)
    elif args.plot_type == "heatmap":
        plot_heatmap(shap_data, args.output, args.max_display)
    elif args.plot_type == "interaction_heatmap":
        plot_interaction_heatmap(shap_data, args.output, args.max_display)
    elif args.plot_type == "waterfall":
        plot_waterfall(shap_data, args.sample_index, args.output)
    elif args.plot_type == "force":
        plot_force(shap_data, args.sample_index, args.output)
    elif args.plot_type == "decision":
        plot_decision(shap_data, args.sample_index, args.output)
    elif args.plot_type == "interaction_scatter":
        plot_interaction_scatter(shap_data, args.output)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot_type}")


if __name__ == "__main__":
    main()

