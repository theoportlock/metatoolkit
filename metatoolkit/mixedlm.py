#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run mixed-effects models on many response variables.")
    parser.add_argument("--formula", type=str, required=True, help="Model formula (e.g., ~ timepoint * Feed + Sex)")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to metadata TSV file (index_col=0)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to data TSV file (index_col=0)")
    parser.add_argument("--grouping_variable", type=str, required=True, help="Column name in metadata used for grouping (random effect)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output TSV for model results")
    return parser.parse_args()

def fit_mixed_model(y, metadata, formula, group_col):
    data = metadata.copy()
    data["__response__"] = y
    try:
        model = smf.mixedlm(f"__response__ {formula}", data, groups=data[group_col])
        result = model.fit(reml=False)
        summary = result.summary().tables[1]
        summary = summary.reset_index().rename(columns={"index": "term"})
        summary["status"] = "OK"
        return summary[["term", "Coef.", "P>|z|", "Std.Err.", "status"]]
    except Exception as e:
        return pd.DataFrame([{
            "term": "NA", "Coef.": None, "P>|z|": None, "Std.Err.": None, "status": f"ERROR: {e}"
        }])

def main():
    args = parse_args()
    
    metadata = pd.read_csv(args.metadata_file, sep="\t", index_col=0)
    data = pd.read_csv(args.data_file, sep="\t", index_col=0)
    
    # Align samples
    shared_samples = metadata.index.intersection(data.index)
    metadata = metadata.loc[shared_samples]
    data = data.loc[shared_samples]

    results = []
    for feature in tqdm(data.columns, desc="Modeling features"):
        y = data[feature]
        summary_df = fit_mixed_model(y, metadata, args.formula, args.grouping_variable)
        summary_df.insert(0, "feature", feature)
        results.append(summary_df)

    final_df = pd.concat(results, axis=0)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_df.to_csv(args.output_file, sep="\t", index=False)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()

