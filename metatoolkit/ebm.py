#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
from pathlib import Path

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def build_model(task, args):
    params = {
        "max_bins": args.max_bins,
        "max_interaction_bins": args.max_interaction_bins,
        "interactions": args.interactions,
        "learning_rate": args.learning_rate,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
    }

    if task == "classification":
        return ExplainableBoostingClassifier(**params)
    elif task == "regression":
        return ExplainableBoostingRegressor(**params)
    else:
        raise ValueError(f"Unsupported task type: {task}")


def save_model(model, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an Explainable Boosting Machine (EBM) using pre-split data"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing X_train.tsv and y_train.tsv")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the trained model (.joblib)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True,
                        help="Type of task to perform")
    parser.add_argument("--max_bins", type=int, default=256,
                        help="Maximum number of bins per feature (default: 256)")
    parser.add_argument("--max_interaction_bins", type=int, default=32,
                        help="Maximum bins for interactions (default: 32)")
    parser.add_argument("--interactions", type=int, default=0,
                        help="Number of pairwise interactions to include (default: 0)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--validation_size", type=float, default=0.15,
                        help="Fraction of training data used for validation (default: 0.15)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed (default: 42)")
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

