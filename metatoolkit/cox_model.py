#!/usr/bin/env python

import argparse
import os
import pandas as pd
from lifelines import CoxPHFitter

def parse_args():
    parser = argparse.ArgumentParser(description="Generic Cox Proportional Hazards CLI")
    parser.add_argument("--input", required=True, help="Path to primary data file")
    parser.add_argument("--meta", required=True, help="Path to metadata file")
    parser.add_argument("--index", default="sampleID")
    parser.add_argument("--subject", default="subjectID")
    parser.add_argument("--time", required=True, help="Time column")
    parser.add_argument("--event", required=True, help="Event column (1/0)")
    parser.add_argument("--covariates", nargs='+', required=True, 
                        help="List of columns to include (e.g., Condition Sex Ethnicity)")
    parser.add_argument("--output", help="Path to save coefficients (CSV)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load and Merge
    sep = '\t' if args.input.endswith('.tsv') else ','
    df = pd.merge(pd.read_csv(args.input, sep=sep), 
                  pd.read_csv(args.meta, sep=sep), on=args.index)
    
    # Process to survival format
    def summarize(group):
        occ = group[group[args.event] == 1]
        res = {
            'duration': occ[args.time].min() if not occ.empty else group[args.time].max(),
            'event': 1 if not occ.empty else 0
        }
        # Keep the first value of each covariate for the subject
        for cov in args.covariates:
            res[cov] = group[cov].iloc[0]
        return pd.Series(res)

    surv_df = df.groupby(args.subject).apply(summarize, include_groups=False).reset_index()
    
    # Drop SubjectID for the model, keep only duration, event, and covariates
    model_data = surv_df[['duration', 'event'] + args.covariates].dropna()
    
    # Convert categorical strings to dummy/indicator variables (One-Hot Encoding)
    model_data = pd.get_dummies(model_data, drop_first=True)

    # Fit Cox Model
    cph = CoxPHFitter()
    cph.fit(model_data, duration_col='duration', event_col='event')
    
    print("\n--- Cox Proportional Hazards Model Summary ---")
    cph.print_summary()

    if args.output:
        cph.summary.to_csv(args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
