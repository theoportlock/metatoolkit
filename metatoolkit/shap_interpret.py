#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SHAP and SHAP interaction scores")
    parser.add_argument("--model", type=str, required=True, help="Path to joblib model file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with X_train.tsv and/or X_test.tsv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SHAP results")
    parser.add_argument("--shap_val", action="store_true", help="Compute SHAP values")
    parser.add_argument("--shap_interact", action="store_true", help="Compute SHAP interaction values")
    return parser.parse_args()


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def save_tsv(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)


def create_explainer(model, output_dir, label):
    explainer = shap.Explainer(model, seed=42)
    explainer_path = os.path.join(output_dir, f"explainer_{label}.joblib")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(explainer, explainer_path)
    print(f"Saved SHAP explainer to {explainer_path}")
    return explainer


def is_classifier(model):
    return hasattr(model, "predict_proba")


def compute_shap_values(explainer, X, label, output_dir, is_classification):
    shap_values = explainer(X)

    # Handle classification: select class 1 SHAP values
    if is_classification:
        if isinstance(shap_values.values, list):
            shap_class_values = shap_values.values[1]
        elif shap_values.values.ndim == 3 and shap_values.values.shape[1] == 2:
            shap_class_values = shap_values.values[:, 1, :]
        else:
            shap_class_values = shap_values.values
    else:
        shap_class_values = shap_values.values

    shap_path = os.path.join(output_dir, f"shap_values_{label}.joblib")
    joblib.dump(shap_values, shap_path)
    print(f"Saved SHAP values to {shap_path}")

    shap_mean = np.abs(shap_class_values).mean(axis=0)

    if shap_mean.ndim == 1:
        mean_abs_shap = pd.DataFrame(
            shap_mean,
            index=X.columns,
            columns=[f"{label}_mean_abs_shap"]
        )
    elif shap_mean.ndim == 2:
        class_labels = [f"class_{i}" for i in range(shap_mean.shape[1])]
        mean_abs_shap = pd.DataFrame(
            shap_mean,
            index=X.columns,
            columns=[f"{label}_mean_abs_shap_{cls}" for cls in class_labels]
        )
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_mean.shape}")

    mean_abs_path = os.path.join(output_dir, f"mean_abs_shap_{label}.tsv")
    save_tsv(mean_abs_shap.reset_index().rename(columns={"index": "feature"}), mean_abs_path)
    print(f"Saved {mean_abs_path}")

def compute_shap_interactions(explainer, X, label, output_dir, is_classification):
    try:
        raw_shap_inter = explainer.shap_interaction_values(X)

        if is_classification:
            if isinstance(raw_shap_inter, list):
                # If it's a list, select the values for the positive class (class 1)
                shap_inter = raw_shap_inter[1]
            elif raw_shap_inter.ndim == 4 and raw_shap_inter.shape[1] == 2:
                # This handles the (n_samples, n_classes, n_features, n_features) case
                shap_inter = raw_shap_inter[:, 1, :, :]
            elif raw_shap_inter.ndim == 4 and raw_shap_inter.shape[3] == 2:
                # THIS IS THE NEW LOGIC for the (n_samples, n_features, n_features, n_classes) case
                shap_inter = raw_shap_inter[:, :, :, 1]
            else:
                raise ValueError(f"Unexpected SHAP interaction shape for classification: {np.shape(raw_shap_inter)}")
        else:
            shap_inter = raw_shap_inter

        # After slicing/selecting, the interaction values should be a 3D array
        # of shape (n_samples, n_features, n_features)
        if shap_inter.ndim != 3:
            raise ValueError(f"Expected 3D SHAP interaction array after processing, but got shape: {np.shape(shap_inter)}")

        # Compute the mean absolute interaction value for each feature pair
        mean_abs_inter = np.abs(shap_inter).mean(axis=0)

        # Create a DataFrame from the resulting square matrix
        inter_df = pd.DataFrame(mean_abs_inter, index=X.columns, columns=X.columns)

        os.makedirs(output_dir, exist_ok=True)

        full_inter_path = os.path.join(output_dir, f"shap_interaction_values_{label}.joblib")
        joblib.dump(shap_inter, full_inter_path)
        print(f"Saved SHAP interaction values to {full_inter_path}")

        long_df = inter_df.stack().reset_index()
        long_df.columns = ["feature1", "feature2", "mean_abs_weight"]
        long_df[["source", "target"]] = np.sort(long_df[["feature1", "feature2"]].values, axis=1)

        grouped = (
            long_df.groupby(["source", "target"], as_index=False)
                   .agg({"mean_abs_weight": "sum"})
        )

        total = grouped["mean_abs_weight"].sum()
        grouped["relative_importance"] = grouped["mean_abs_weight"] / total if total > 0 else 0.0

        output_file = os.path.join(output_dir, f"mean_abs_shap_interaction_{label}.tsv")
        save_tsv(grouped, output_file)
        print(f"Saved SHAP interaction file to: {output_file}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Could not compute SHAP interaction values: {e}")


def main():
    args = parse_args()
    model = joblib.load(args.model)
    is_classification = is_classifier(model)

    for split in ["train", "test"]:
        x_path = Path(args.input_dir) / f"X_{split}.tsv"
        if not x_path.exists():
            print(f"Skipping: {x_path} not found")
            continue

        X = load_tsv(x_path)
        explainer = create_explainer(model, args.output_dir, label=split)

        if args.shap_val:
            compute_shap_values(
                explainer, X, label=split,
                output_dir=args.output_dir,
                is_classification=is_classification
            )

        if args.shap_interact:
            compute_shap_interactions(
                explainer, X, label=split,
                output_dir=args.output_dir,
                is_classification=is_classification
            )


if __name__ == "__main__":
    main()

