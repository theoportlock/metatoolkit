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
        description="General Linear Mixed Effects Models (LMM) for multiple response variables."
    )
    parser.add_argument("-i", "--response_data", required=True,
                        help="Input TSV/CSV with samples as rows and response variables as columns (e.g., genes, metabolites, diversity metrics).")
    parser.add_argument("-m", "--metadata", required=True,
                        help="Metadata TSV/CSV file containing predictors and grouping variables (index_col=0).")
    parser.add_argument("-f", "--formula", required=True,
                        help="Formula Right-Hand Side (e.g. 'Timepoint * Treatment + Age'). Do not include the dependent variable name.")
    parser.add_argument("-g", "--grouping_variable", required=True,
                        help="Column name for the Random Effect (e.g. SubjectID, PatientID).")
    parser.add_argument("-o", "--output", required=True, help="Output filename for results (TSV).")
    return parser.parse_args()

def sanitize_colnames(df):
    """
    Statsmodels/Patsy struggles with column names containing '.', spaces, or starting with numbers.
    We replace '.' and spaces with '_' to ensure formulas work.
    """
    new_cols = {col: col.replace('.', '_').replace(' ', '_').replace('-', '_') for col in df.columns}
    return df.rename(columns=new_cols), new_cols

def fit_mixed_model(y, metadata, formula, group_col):
    # Combine response and metadata
    df = metadata.copy()
    # We use a generic internal name for the response variable to simplify the formula injection
    target_col_name = "__target_response__"
    df[target_col_name] = y

    # Ensure numeric data is numeric
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors='coerce')

    # Drop NaN values specifically for the columns involved in this model
    # We extract variables from formula to check for NaNs
    formula_vars = [x.strip() for x in re.split(r'[+*():]', formula) if x.strip()]

    # Clean out interaction terms or function calls from the variable list for NaN checking
    # This removes things like "C(Var)" or "np.log(Var)" to just get "Var" ideally,
    # though simple splitting works for 90% of cases.
    clean_vars = []
    for v in formula_vars:
        # Remove patsy wrappers like C(), I(), etc if simple
        v_clean = re.sub(r'^[A-Z]+\((.*)\)$', r'\1', v)
        if v_clean in df.columns:
            clean_vars.append(v_clean)
        elif v in df.columns:
            clean_vars.append(v)

    cols_to_check = [group_col, target_col_name] + clean_vars

    # Filter df to only existing columns to avoid errors if parsing failed
    cols_to_check = [c for c in cols_to_check if c in df.columns]

    df_clean = df.dropna(subset=cols_to_check)

    if df_clean.empty:
        return pd.DataFrame([{
            "term": "NA", "coef": np.nan, "pval": np.nan,
            "status": "ERROR: No data after dropping NaNs"
        }])

    try:
        # Fit Model
        full_formula = f"{target_col_name} ~ {formula}"

        model = smf.mixedlm(full_formula, df_clean, groups=df_clean[group_col])
        result = model.fit(reml=False)

        # Extract Results
        results_df = pd.DataFrame({
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z_score": result.tvalues.values,
            "pval": result.pvalues.values,
            "conf_lower": result.conf_int()[0].values,
            "conf_upper": result.conf_int()[1].values
        })

        results_df["status"] = "Converged"
        results_df["n_obs"] = result.nobs
        results_df["n_groups"] = df_clean[group_col].nunique()

        return results_df

    except Exception as e:
        return pd.DataFrame([{
            "term": "NA",
            "coef": np.nan,
            "pval": np.nan,
            "status": f"ERROR: {str(e)}"
        }])

def main():
    args = parse_args()

    # 1. Load Data
    print("Loading data...")
    try:
        # Detect separator based on extension, default to tab
        sep_resp = ',' if args.response_data.endswith('.csv') else '\t'
        sep_meta = ',' if args.metadata.endswith('.csv') else '\t'

        response_df = pd.read_csv(args.response_data, sep=sep_resp, index_col=0)
        metadata = pd.read_csv(args.metadata, sep=sep_meta, index_col=0)
    except Exception as e:
        sys.exit(f"Error reading files: {e}")

    # 2. Sanitize Metadata Column Names
    metadata, name_map = sanitize_colnames(metadata)

    print(f"Sanitized metadata columns (dots/spaces replaced by underscores).")

    # Adjust user formula arguments to match sanitized names
    # CRITICAL FIX: Only replace dots in formula. Do NOT replace spaces or hyphens as they are syntax.
    sanitized_formula = args.formula.replace('.', '_')

    # Grouping variable IS a column name, so it needs full sanitization
    sanitized_group = args.grouping_variable.replace('.', '_').replace(' ', '_').replace('-', '_')

    print(f"Using formula: Response ~ {sanitized_formula}")
    print(f"Grouping by: {sanitized_group}")

    # 3. Align samples
    shared_samples = response_df.index.intersection(metadata.index)
    if len(shared_samples) == 0:
        sys.exit("Error: No matching sample IDs found between response data and metadata files.")

    print(f"Found {len(shared_samples)} shared samples.")
    response_df = response_df.loc[shared_samples]
    metadata = metadata.loc[shared_samples]

    results = []

    # 4. Iterate and Model
    # We iterate over every column in the response dataframe
    for feature_name in tqdm(response_df.columns, desc="Fitting mixed models"):
        y = response_df[feature_name]

        # Skip non-numeric columns in response data
        if not pd.api.types.is_numeric_dtype(y):
            continue

        res = fit_mixed_model(y, metadata, sanitized_formula, sanitized_group)
        res.insert(0, "response_variable", feature_name)
        results.append(res)

    # 5. Save
    if not results:
        sys.exit("No models were successfully run. Check if your input data is numeric.")

    final = pd.concat(results, axis=0)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final.to_csv(args.output, sep="\t", index=False)

    print(f"✅ Results saved to: {args.output}")

if __name__ == "__main__":
    main()
