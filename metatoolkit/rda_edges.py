#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
from skbio.stats.ordination import rda
from sklearn.preprocessing import StandardScaler
#DOESNT WORK


def parse_args():
    parser = argparse.ArgumentParser(description="Perform RDA and summarize variance components")
    parser.add_argument("-r", "--response_data", required=True, help="Path to response data file (TSV)")
    parser.add_argument("-m", "--metadata", required=True, help="Path to metadata file (TSV)")
    parser.add_argument("-f", "--fixed_effects", default=None,
                        help="Comma-separated list of fixed effects (e.g. 'Factor1,Factor2'). If not provided, use all metadata columns.")
    parser.add_argument("-o", "--output", default="results/rda_variance.tsv", help="Path to output TSV file")
    return parser.parse_args()


def load_data(response_path, metadata_path):
    response = pd.read_csv(response_path, sep="\t", index_col=0)
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    common_samples = response.index.intersection(metadata.index)
    response = response.loc[common_samples]
    metadata = metadata.loc[common_samples]
    return response, metadata


def run_rda(Y, X):
    return rda(Y.values, X.values, scale_Y=False, scaling=2)


def compute_explained_variance(response, metadata, effects):
    Y = response
    full_X = pd.get_dummies(metadata[effects], drop_first=True)
    full_X = full_X.loc[Y.index]

    rda_result = run_rda(Y, full_X)
    total_constrained = rda_result.proportion_explained.sum()
    residual_var = 1.0 - total_constrained

    explained_rows = []

    for effect in effects:
        X_effect = pd.get_dummies(metadata[[effect]].loc[Y.index], drop_first=True)
        rda_sub = run_rda(Y, X_effect)
        effect_var = rda_sub.proportion_explained.sum()
        explained_rows.append({
            "source": effect,
            "target": "Response variation",
            "variance": effect_var,
            "significance": "NA"
        })

    explained_rows.append({
        "source": "Residual",
        "target": "Response variation",
        "variance": residual_var,
        "significance": "NA"
    })

    # Normalize to sum to 1
    total_var = sum(row["variance"] for row in explained_rows)
    for row in explained_rows:
        row["variance"] /= total_var

    return pd.DataFrame(explained_rows)


def save_variance_table(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)


def main():
    args = parse_args()
    response, metadata = load_data(args.response_data, args.metadata)

    if args.fixed_effects:
        effects = args.fixed_effects.split(',')
    else:
        effects = metadata.columns.tolist()

    variance_df = compute_explained_variance(response, metadata, effects)
    save_variance_table(variance_df, args.output)


if __name__ == "__main__":
    main()

