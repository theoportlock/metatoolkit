#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)

def build_model(task, args):
    # Common hyperparameters
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    if task == "classification":
        return RandomForestClassifier(**params)
    elif task == "regression":
        return RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unsupported task type: {task}")
    return model


def save_model(model, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a RandomForest model (classifier or regressor) using pre-split data"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing X_train.tsv and y_train.tsv")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the trained model (.joblib)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True,
                        help="Type of task to perform (classification or regression)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the forest (default: 100)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of the tree (default: None)")
    parser.add_argument("--min_samples_split", type=int, default=2,
                        help="Minimum samples to split an internal node (default: 2)")
    parser.add_argument("--min_samples_leaf", type=int, default=1,
                        help="Minimum samples required at a leaf node (default: 1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_train = load_tsv(input_dir / "X_train.tsv")
    y_train = load_tsv(input_dir / "y_train.tsv")

    model = build_model(args.task, args)
    model.fit(X_train, y_train.values.ravel())
    save_model(model, args.output_model)


if __name__ == "__main__":
    main()

