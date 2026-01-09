#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="General Linear / Linear Mixed Models for multiple response variables."
    )
    parser.add_argument(
        "-i", "--response_data", required=True,
        help="Input TSV/CSV with samples as rows and response variables as columns."
    )
    parser.add_argument(
        "-m", "--metadata", required=True,
        help="Metadata TSV/CSV file containing predictors (index_col=0)."
    )
    parser.add_argument(
        "-f", "--formula", required=False,
        help="Right-hand side formula. If omitted, uses ALL metadata columns: col1 + col2 + ..."
    )
    parser.add_argument(
        "-g", "--grouping_variable", required=False,
        help="Optional Random Effect grouping variable. If omitted, uses OLS."
    )
    parser.add_argument("-o", "--output", required=True, help="Output filename (TSV).")
    return parser.parse_args()


def sanitize_colnames(df):
    new_cols = {col: col.replace('.', '_').replace(' ', '_').replace('-', '_') for col in df.columns}
    return df.rename(columns=new_cols), new_cols


def build_default_formula(metadata):
    """Construct RHS formula from all metadata columns."""
    cols = metadata.columns
    rhs = " + ".join(cols)
    return rhs


def fit_model(y, metadata, formula, group_col):
    df = metadata.copy()
    target_col_name = "__target_response__"
    df[target_col_name] = y
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors="coerce")

    # Extract predictor names for NaN filtering
    formula_vars = [x.strip() for x in re.split(r"[+*():]", formula) if x.strip()]

    clean_vars = []
    for v in formula_vars:
        v_clean = re.sub(r"^[A-Z]+\((.*)\)$", r"\1", v)
        if v_clean in df.columns:
            clean_vars.append(v_clean)
        elif v in df.columns:
            clean_vars.append(v)

    cols_to_check = [target_col_name] + clean_vars
    if group_col and group_col in df.columns:
        cols_to_check.append(group_col)

    df_clean = df.dropna(subset=[c for c in cols_to_check if c in df.columns])

    if df_clean.empty:
        return pd.DataFrame([{
            "term": "NA", "coef": np.nan, "pval": np.nan,
            "status": "ERROR: No data after dropping NaNs"
        }])

    full_formula = f"{target_col_name} ~ {formula}"

    try:
        # OLS if group_col is None
        if group_col is None:
            result = smf.ols(full_formula, df_clean).fit()
            model_type = "OLS"
        else:
            model = smf.mixedlm(full_formula, df_clean, groups=df_clean[group_col])
            result = model.fit(reml=False)
            model_type = "LMM"

        results_df = pd.DataFrame({
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z_score": result.tvalues,
            "pval": result.pvalues,
            "conf_lower": result.conf_int()[0],
            "conf_upper": result.conf_int()[1],
        })

        results_df["status"] = f"Converged ({model_type})"
        results_df["n_obs"] = result.nobs
        results_df["n_groups"] = (
            df_clean[group_col].nunique() if group_col else np.nan
        )

        return results_df

    except Exception as e:
        return pd.DataFrame([{
            "term": "NA",
            "coef": np.nan,
            "pval": np.nan,
            "status": f"ERROR: {str(e)}"
        }])


def main():
    import sys; print(sys.argv)
    args = parse_args()

    # Load data
    sep_resp = ',' if args.response_data.endswith('.csv') else '\t'
    sep_meta = ',' if args.metadata.endswith('.csv') else '\t'

    try:
        response_df = pd.read_csv(args.response_data, sep=sep_resp, index_col=0)
        metadata = pd.read_csv(args.metadata, sep=sep_meta, index_col=0)
    except Exception as e:
        sys.exit(f"Error reading input files: {e}")

    # Sanitize metadata column names
    metadata, name_map = sanitize_colnames(metadata)

    # Auto-build formula if missing
    if args.formula:
        sanitized_formula = args.formula.replace('.', '_')
    else:
        sanitized_formula = build_default_formula(metadata)
        print(f"No formula supplied → Using all metadata columns:\n  {sanitized_formula}")

    # Optional grouping variable
    if args.grouping_variable:
        sanitized_group = (
            args.grouping_variable.replace('.', '_').replace(' ', '_').replace('-', '_')
        )
        print(f"Using Random Effect grouping variable: {sanitized_group}")
    else:
        sanitized_group = None
        print("No grouping variable supplied → Models will be OLS")

    # Align samples
    shared_samples = response_df.index.intersection(metadata.index)
    if not len(shared_samples):
        sys.exit("No matching sample IDs found between response data and metadata.")

    response_df = response_df.loc[shared_samples]
    metadata = metadata.loc[shared_samples]

    results = []

    # Fit models
    for feature_name in tqdm(response_df.columns, desc="Fitting models"):
        y = response_df[feature_name]

        if not pd.api.types.is_numeric_dtype(y):
            continue

        res = fit_model(y, metadata, sanitized_formula, sanitized_group)
        res.insert(0, "response_variable", feature_name)
        results.append(res)

    if not results:
        sys.exit("No models successfully run.")

    final = pd.concat(results, axis=0)
    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    final.to_csv(args.output, sep="\t", index=False)

    print(f"✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()

