#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_dataset(file_path):
    return pd.read_csv(file_path, sep='\t', index_col=0)


def save_tsv(df, output_dir, filename):
    output_path = Path(output_dir) / f"{filename}.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t')


def get_scaler(scaler_name):
    if scaler_name == "standard":
        return StandardScaler()
    elif scaler_name == "minmax":
        return MinMaxScaler()
    elif scaler_name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scaler option: {scaler_name}")


def is_classification_target(y):
    return (
        pd.api.types.is_categorical_dtype(y)
        or y.dtype == object
        or y.nunique() < 20
    )


def merge_meta(df, meta_paths):
    """Inner-join metadata TSV(s) with main dataframe on the index."""
    for mpath in meta_paths:
        mdf = pd.read_csv(mpath, sep='\t', index_col=0)
        df = df.join(mdf, how='inner')
    return df


def split_and_balance(input_path, output_dir, y_col, test_size, random_state,
                      apply_smote, scaler_name, y_file=None, meta_files=None):
    df = load_dataset(input_path)

    # Merge metadata if provided
    if meta_files:
        df = merge_meta(df, meta_files)

    # Handle separate y_file
    if y_file:
        df2 = load_dataset(y_file)
        if y_col not in df2.columns:
            raise ValueError(f"Column '{y_col}' not found in y_file '{y_file}'.")
        y = df2[y_col]
    else:
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in input data.")
        y = df[y_col]
        df = df.drop(columns=[y_col])

    # Align X and y on index
    Xy = df.join(y, how='inner')

    # Warn about missing values and drop them
    if Xy.isna().any().any():
        missing_count = Xy.isna().sum().sum()
        print(f"WARNING: Found {missing_count} missing values. Dropping all samples with missing values.")
        Xy = Xy.dropna()

    # Split back into X and y
    y = Xy[y_col]
    X_scaled = Xy.drop(columns=[y_col])

    classification = is_classification_target(y)
    if classification and y.dtype == object:
        y = y.astype("category")

    if apply_smote and not classification:
        raise ValueError("SMOTE can only be used with classification (categorical y).")

    scaler = get_scaler(scaler_name)
    if scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_scaled), columns=X_scaled.columns, index=X_scaled.index
        )

    stratify = y if classification else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if apply_smote:
        smoter = SMOTE(random_state=random_state)
        X_train, y_train = smoter.fit_resample(X_train, y_train)

    y_train_df = pd.DataFrame(y_train, columns=[y_col], index=X_train.index)
    y_test_df = pd.DataFrame(y_test, columns=[y_col], index=X_test.index)

    save_tsv(X_train, output_dir, "X_train")
    save_tsv(X_test, output_dir, "X_test")
    save_tsv(y_train_df, output_dir, "y_train")
    save_tsv(y_test_df, output_dir, "y_test")

    print(f"Saved train/test splits to: {output_dir}")
    print(f"Task type: {'classification' if classification else 'regression'}")
    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Scaler used: {scaler_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train/test splitter with optional SMOTE, scaling, metadata merge, and missing value handling"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input TSV file (features)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save X_train, X_test, y_train, y_test")
    parser.add_argument("--y_col", type=str, required=True,
                        help="Name of the target column to predict")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of the data to use for testing (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--smote", action="store_true",
                        help="Apply SMOTE to balance training data (only for classification)")
    parser.add_argument("--scaler", type=str, default="standard",
                        choices=["standard", "minmax", "none"],
                        help="Feature scaling method (default: standard)")
    parser.add_argument("--y_file", type=str, default=None,
                        help="Optional path to TSV file containing y_col to join on index")
    parser.add_argument("--meta", nargs='+', default=None,
                        help="Optional path(s) to metadata TSV file(s) to inner-join with input data")

    args = parser.parse_args()

    split_and_balance(
        input_path=args.input,
        output_dir=args.output_dir,
        y_col=args.y_col,
        test_size=args.test_size,
        random_state=args.random_state,
        apply_smote=args.smote,
        scaler_name=args.scaler,
        y_file=args.y_file,
        meta_files=args.meta,
    )


if __name__ == "__main__":
    main()

