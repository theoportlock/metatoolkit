#!/usr/bin/env python3
"""
Create a nested donut (sunburst-like) plot of SHAP values.

Usage:
    python sunburst_matplotlib.py \
        --input results/prediction/dataset_rf_shap/mean_abs_shap_test.tsv \
        --output sunburst_shap_matplotlib.svg \
        --threshold 0.01
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def lighten(color, alpha=0.4):
    """Return a lighter / transparent version of a color."""
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def plot_nested_donut(df: pd.DataFrame, output_file: str, threshold: float = 0.0):
    """
    Nested donut:
    - Inner ring: categories
    - Outer ring: all features (keep arc), but
      features < threshold are drawn in white and have no label.
    """
    # Category totals
    cat_total = df.groupby("category")["test_mean_abs_shap"].sum()
    cat_order = cat_total.sort_values(ascending=False)

    # Base colors for each category
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cat_colors = {cat: base_colors[i % len(base_colors)]
                  for i, cat in enumerate(cat_order.index)}

    inner_sizes, inner_labels, inner_colors = [], [], []
    outer_sizes, outer_labels, outer_colors = [], [], []

    for cat in cat_order.index:
        # Inner ring (categories)
        inner_sizes.append(cat_total[cat])
        inner_labels.append(f"{cat}\n({cat_total[cat]/cat_total.sum()*100:.1f}%)")
        inner_colors.append(cat_colors[cat])

        # Outer ring: keep every feature, but mask those < threshold
        sub = (df[df["category"] == cat]
               .sort_values("test_mean_abs_shap", ascending=False))
        for _, row in sub.iterrows():
            outer_sizes.append(row["test_mean_abs_shap"])
            if row["test_mean_abs_shap"] >= threshold:
                outer_labels.append(row["feature"])
                outer_colors.append(lighten(cat_colors[cat], 0.5))
            else:
                outer_labels.append("")      # no text
                outer_colors.append("white") # white wedge

    # --------- Plot ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')

    ax.pie(outer_sizes, radius=1, labels=outer_labels,
           labeldistance=1.05, colors=outer_colors,
           wedgeprops=dict(width=0.3, edgecolor='w'))

    ax.pie(inner_sizes, radius=0.7, labels=inner_labels,
           labeldistance=0.5, colors=inner_colors,
           wedgeprops=dict(width=0.3, edgecolor='w'))

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")



# -------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Nested donut of SHAP values.")
    parser.add_argument("--input", required=True,
                        help="Input TSV with SHAP values.")
    parser.add_argument("--output", required=True,
                        help="Output SVG (or PNG).")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Hide features whose SHAP value is below this number.")
    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()

    # Load data and split category/feature
    df = (
        pd.read_csv(args.input, sep="\t", index_col=0)
          .rename_axis("cat_feat")
    )
    df.index = df.index.str.split(":", expand=True)
    df = df.reset_index().set_axis(["category", "feature", "test_mean_abs_shap"], axis=1)

    plot_nested_donut(df, args.output, threshold=args.threshold)



if __name__ == "__main__":
    main()

