#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate residual plots for a regression model"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.joblib)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing X_test.tsv and y_test.tsv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save residual plots (SVG format)")
    return parser


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def plot_residuals_and_diagnostics(y_true, y_pred, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    # === Plot 1: Residuals vs Predicted ===
    plt.figure(figsize=(3, 3))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_predicted.svg", format="svg")
    plt.close()

    # === Plot 2: Histogram of Residuals ===
    plt.figure(figsize=(3, 3))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_distribution.svg", format="svg")
    plt.close()

    # === Plot 3: Observed vs Predicted (with R²) ===
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(3, 3))
    sns.regplot(x=y_true, y=y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.6})
    plt.title(f"Observed vs Predicted (R² = {r2:.3f})")
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_vs_predicted.svg", format="svg")
    plt.close()

    # === Plot 4: Q-Q Plot ===
    plt.figure(figsize=(3, 3))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "qq_plot.svg", format="svg")
    plt.close()

    print(f"SVG plots saved to: {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    model = joblib.load(args.model)
    X_test = load_tsv(Path(args.input_dir) / "X_test.tsv")
    y_test = load_tsv(Path(args.input_dir) / "y_test.tsv").squeeze()

    y_pred = model.predict(X_test)

    plot_residuals_and_diagnostics(y_test, y_pred, args.output_dir)


if __name__ == "__main__":
    main()

