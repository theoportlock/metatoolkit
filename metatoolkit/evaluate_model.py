#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on test data (classification or regression)"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.joblib)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing X_test.tsv and y_test.tsv")
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "regression"],
                        help="Type of ML task: classification or regression")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Path to save evaluation report (.tsv)")
    return parser


def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def evaluate(model_path, input_dir, task_type, report_file):
    input_dir = Path(input_dir)

    model = joblib.load(model_path)
    X_test = load_tsv(input_dir / "X_test.tsv")
    y_test = load_tsv(input_dir / "y_test.tsv")

    y_pred = model.predict(X_test)

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if task_type == "classification":
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        print("≡ƒÄä Classification Report:")
        print(report_df)
        report_df.to_csv(report_path, sep='\t')
    elif task_type == "regression":
        metrics = {
            "mean_squared_error": [mean_squared_error(y_test, y_pred)],
            "mean_absolute_error": [mean_absolute_error(y_test, y_pred)],
            "r2_score": [r2_score(y_test, y_pred)],
        }
        report_df = pd.DataFrame(metrics)
        print("Regression Metrics:")
        print(report_df)
        report_df.to_csv(report_path, sep='\t', index=False)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    print(f"Report saved to: {report_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args.model, args.input_dir, args.task, args.report_file)


if __name__ == "__main__":
    main()

