#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def build_xgb_model(task, args):
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    if task == "classification":
        return XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    elif task == "regression":
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported task: {task}")


def save_model(model, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"✅ Model saved to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model (classifier or regressor) using pre-split data"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing X_train.tsv and y_train.tsv")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the trained model (.joblib)")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True,
                        help="Task type")

    # XGBoost hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of boosting rounds")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate (eta)")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Subsample ratio of training instances")
    parser.add_argument("--colsample_bytree", type=float, default=1.0,
                        help="Subsample ratio of columns when constructing each tree")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    X_train = load_tsv(input_dir / "X_train.tsv")
    y_train = load_tsv(input_dir / "y_train.tsv")

    model = build_xgb_model(args.task, args)
    model.fit(X_train, y_train.values.ravel())
    save_model(model, args.output_model)


if __name__ == "__main__":
    main()

