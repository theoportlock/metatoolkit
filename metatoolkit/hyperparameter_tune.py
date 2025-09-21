#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hyperparameter_tune.py
Theo Portlock

Hyperparameter tuning wrapper that reuses:
- build_model() from random_forest.py
- evaluate() from evaluate_model.py
"""

import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, accuracy_score

# Import your scripts
import random_forest
import evaluate_model


def load_split_data(input_dir):
    """Assumes input_dir has X_train.tsv, y_train.tsv"""
    X_train = pd.read_csv(Path(input_dir) / "X_train.tsv", sep="\t", index_col=0)
    y_train = pd.read_csv(Path(input_dir) / "y_train.tsv", sep="\t", index_col=0).squeeze()
    return X_train, y_train


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning using GridSearchCV or RandomizedSearchCV")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with train/test splits")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--search_grid", type=str, required=True, help="JSON file with parameter grid")
    parser.add_argument("--output_model", type=str, required=True, help="File to save best model (pkl)")
    parser.add_argument("--report_file", type=str, required=True, help="TSV file to save evaluation report")
    parser.add_argument("--search", choices=["grid", "random"], default="grid", help="Search strategy")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for random search")
    args = parser.parse_args()

    # Load training data
    X_train, y_train = load_split_data(args.input_dir)

    # Load param grid
    with open(args.search_grid) as f:
        param_grid = json.load(f)

    # Base args for RF
    base_args = argparse.Namespace(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        task=args.task,
    )

    # Build model
    model = random_forest.build_model(args.task, base_args)

    # CV scorer
    if args.task == "regression":
        scorer = make_scorer(r2_score)
    else:
        scorer = make_scorer(accuracy_score)

    # Search
    if args.search == "grid":
        search = GridSearchCV(
            model,
            param_grid,
            scoring=scorer,
            cv=args.cv,
            n_jobs=-1,
            verbose=1,
        )
    else:
        search = RandomizedSearchCV(
            model,
            param_grid,
            scoring=scorer,
            cv=args.cv,
            n_iter=args.n_iter,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

    # Fit
    search.fit(X_train, y_train)

    # Save best model
    best_model = search.best_estimator_
    joblib.dump(best_model, args.output_model)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    print(f"Model saved to: {args.output_model}")

    # Save CV results as extra info
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["best"] = cv_results["rank_test_score"] == 1
    cv_results.to_csv(Path(args.report_file).with_suffix(".cv.tsv"), sep="\t", index=False)

    # Run your full evaluation on test set
    evaluate_model.evaluate(args.output_model, args.input_dir, args.task, args.report_file)


if __name__ == "__main__":
    main()

